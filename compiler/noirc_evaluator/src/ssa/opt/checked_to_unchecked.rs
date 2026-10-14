//! This SSA pass will turn checked unsigned binary additions, subtractions and multiplications
//! into unchecked ones if it's guaranteed that the operations cannot overflow.
//!
//! Signed checked binary operations should have already been converted to unchecked ones with
//! an explicit overflow check during [`super::expand_signed_checks`].
use acvm::{AcirField as _, FieldElement};
use rustc_hash::FxHashMap as HashMap;

use crate::ssa::{
    ir::{
        dfg::DataFlowGraph,
        function::Function,
        instruction::{Binary, BinaryOp, Instruction},
        types::NumericType,
        value::{Value, ValueId},
    },
    ssa_gen::Ssa,
};

impl Ssa {
    /// See [`checked_to_unchecked`][self] module for more information.
    #[tracing::instrument(level = "trace", skip(self))]
    pub(crate) fn checked_to_unchecked(mut self) -> Ssa {
        for function in self.functions.values_mut() {
            function.checked_to_unchecked();
        }
        self
    }
}

impl Function {
    fn checked_to_unchecked(&mut self) {
        #[cfg(debug_assertions)]
        checked_to_unchecked_pre_check(self);

        let mut value_max_num_bits = HashMap::<ValueId, u32>::default();

        self.simple_optimization(|context| {
            let instruction = context.instruction();
            let Instruction::Binary(binary) = instruction else {
                return;
            };
            let lhs = binary.lhs;
            let rhs = binary.rhs;

            let lhs_type = context.dfg.type_of_value(lhs).unwrap_numeric();
            let NumericType::Unsigned { .. } = lhs_type else {
                return;
            };

            let dfg = &context.dfg;

            let unchecked = match binary.operator {
                BinaryOp::Add { unchecked: false } => {
                    let bit_size = dfg.type_of_value(lhs).bit_size();
                    let max_lhs_bits = required_bit_size(dfg, lhs, &mut value_max_num_bits);
                    let max_rhs_bits = required_bit_size(dfg, rhs, &mut value_max_num_bits);

                    // 1. If both lhs and rhs have less max bits than the result it means their
                    //    value is at most `2^(n-1) - 1`, assuming `n = bit_size`. Adding those
                    //    we get `2^(n-1) - 1 + 2^(n-1) - 1`, so `2*(2^(n-1)) - 2`,
                    //    so `2^n - 2` which fits in `0..2^n`.
                    // In that case, `lhs` and `rhs` have both been casted up from smaller types and so cannot overflow.
                    max_lhs_bits < bit_size && max_rhs_bits < bit_size
                }
                BinaryOp::Sub { unchecked: false } => {
                    // True when an unsigned subtraction `lhs - rhs` is guaranteed not to underflow.
                    //
                    // This is the case when `lhs` is a constant that is >= the maximum possible value of `rhs`
                    // (determined by its bit width). For example, `256 - (x as u32)` where `x: u8` cannot
                    // underflow because `256 >= 255`.

                    if let Some(lhs_const) = dfg.get_numeric_constant(lhs) {
                        let max_rhs_bits = required_bit_size(dfg, rhs, &mut value_max_num_bits);
                        // `rhs` may need more bits than its type if it is unchecked ACIR
                        // arithmetic; a u128 bound cannot be exceeded by any constant here.
                        if max_rhs_bits > 128 {
                            false
                        } else {
                            let max_rhs = if max_rhs_bits == 128 {
                                u128::MAX
                            } else {
                                (1 << max_rhs_bits) - 1
                            };

                            // `lhs` is a fixed constant and `rhs` is restricted such that `lhs - rhs >= 0`.
                            // For example: `lhs` is 1 and `rhs` max bitsize is 1, so at most it's `1 - 1`.
                            // Another example: `lhs` is 255 and `rhs` max bitsize is 8, so at most it's `255 - 255`.
                            lhs_const >= max_rhs.into()
                        }
                    } else {
                        false
                    }
                }
                BinaryOp::Mul { unchecked: false } => {
                    let bit_size = dfg.type_of_value(lhs).bit_size();
                    let max_lhs_bits = required_bit_size(dfg, lhs, &mut value_max_num_bits);
                    let max_rhs_bits = required_bit_size(dfg, rhs, &mut value_max_num_bits);

                    // `required_bit_size` tracks the actual range of a value through casts,
                    // truncations, and boolean multiplications — it may be smaller than the
                    // type's bit_size (e.g. a u8 upcast to u64 still has max_bits == 8).
                    //
                    // The product of an `a`-bit value and a `b`-bit value needs at most
                    // `a + b` bits: `(2^a - 1) * (2^b - 1) < 2^(a+b)`. So if
                    // `max_lhs_bits + max_rhs_bits <= bit_size`, the result is guaranteed
                    // to fit and the multiplication cannot overflow.
                    //
                    // As a special case, when either operand has `max_bits == 1` its value
                    // is at most 1, so `x * 0 = 0` or `x * 1 = x` — neither can overflow as
                    // long as `x` fits the type, which in ACIR an unchecked result may not.
                    // This is sound as long as `required_bit_size` never returns 1 for a
                    // value that could actually exceed 1.
                    max_lhs_bits + max_rhs_bits <= bit_size
                        || (max_lhs_bits == 1 && max_rhs_bits <= bit_size)
                        || (max_rhs_bits == 1 && max_lhs_bits <= bit_size)
                }
                _ => false,
            };
            if unchecked {
                let operator = binary.operator.into_unchecked();
                context.replace_current_instruction_with(Instruction::Binary(Binary {
                    lhs: binary.lhs,
                    rhs: binary.rhs,
                    operator,
                }));
            }
        });
    }
}

/// Returns a maximum number of bits the `value` requires.
/// The logic here is almost the same as [`DataFlowGraph::get_value_max_num_bits`] except that
/// - it takes into account that the bitsize of multiplying two bools is 1
/// - it recurses by memoizing the results in `value_max_num_bits`
fn required_bit_size(
    dfg: &DataFlowGraph,
    value: ValueId,
    value_max_num_bits: &mut HashMap<ValueId, u32>,
) -> u32 {
    required_bit_size_at_depth(dfg, value, value_max_num_bits, 0)
}

/// How deep [`required_bit_size`] follows a value's operands before it gives up and returns the
/// widest bound the value can have, so that a long chain of instructions cannot exhaust the stack.
const MAX_DEPTH: usize = 1000;

fn required_bit_size_at_depth(
    dfg: &DataFlowGraph,
    value: ValueId,
    value_max_num_bits: &mut HashMap<ValueId, u32>,
    depth: usize,
) -> u32 {
    if let Some(bits) = value_max_num_bits.get(&value) {
        return *bits;
    }

    let value_bit_size = dfg.type_of_value(value).bit_size();
    let field_bit_size = FieldElement::max_num_bits();
    if depth >= MAX_DEPTH {
        // Not memoized: a shallower query may still find a tighter bound.
        return if dfg.runtime().is_acir() { field_bit_size } else { value_bit_size };
    }
    let mut required =
        |value| required_bit_size_at_depth(dfg, value, value_max_num_bits, depth + 1);

    let bits = match dfg[value] {
        Value::Instruction { instruction, .. } => {
            match dfg[instruction] {
                Instruction::Cast(original_value, _) => {
                    let original_bit_size = required(original_value);
                    // We might have cast e.g. `u1` to `u8` to be able to do arithmetic,
                    // in which case we want to recover the original smaller bit size;
                    // OTOH if we cast down, then we don't need the higher original size.
                    value_bit_size.min(original_bit_size)
                }
                Instruction::Binary(Binary { lhs, operator: BinaryOp::Mul { .. }, rhs })
                    if required(lhs) == 1 && required(rhs) == 1 =>
                {
                    // When multiplying two values, if their bitsize is 1 then the result's bitsize will be 1 too
                    1
                }
                Instruction::Truncate { value, bit_size, .. } => {
                    let value_bit_size = value_bit_size.min(required(value));
                    value_bit_size.min(bit_size)
                }
                // In ACIR, unchecked arithmetic is not reduced to the type width, so its result
                // may need more bits than its type: `unchecked_add u1 1, 1` is the field value 2,
                // and an unchecked Sub can underflow to a field-negative value of any width.
                // Bound it from the operands instead, as `DataFlowGraph::get_value_max_num_bits`
                // does, but following the operands' own bounds. The result may exceed
                // `value_bit_size`. (In Brillig the result wraps to the type width.)
                Instruction::Binary(Binary {
                    lhs,
                    operator:
                        operator @ (BinaryOp::Add { unchecked: true }
                        | BinaryOp::Sub { unchecked: true }
                        | BinaryOp::Mul { unchecked: true }),
                    rhs,
                }) if dfg.runtime().is_acir() => {
                    {
                        match operator {
                            BinaryOp::Add { .. } => {
                                selected_sum_bound(dfg, lhs, rhs, &mut required).unwrap_or_else(
                                    || required(lhs).max(required(rhs)).saturating_add(1),
                                )
                            }
                            BinaryOp::Mul { .. } => {
                                // A 0/1 operand selects the other operand or zero.
                                let lhs_bits = required(lhs);
                                let rhs_bits = required(rhs);
                                if lhs_bits <= 1 {
                                    rhs_bits
                                } else if rhs_bits <= 1 {
                                    lhs_bits
                                } else {
                                    lhs_bits.saturating_add(rhs_bits)
                                }
                            }
                            _ => field_bit_size,
                        }
                        .min(field_bit_size)
                    }
                }
                // In ACIR `not x` is `2^n - 1 - x`, so an `x` wider than its type makes it
                // field-negative.
                Instruction::Not(operand)
                    if dfg.runtime().is_acir() && required(operand) > value_bit_size =>
                {
                    field_bit_size
                }
                _ => value_bit_size,
            }
        }
        Value::NumericConstant { constant, .. } => constant.num_bits(),
        _ => value_bit_size,
    };

    assert!(bits <= value_bit_size || dfg.runtime().is_acir());
    value_max_num_bits.insert(value, bits);

    bits
}

/// The ways `term` can be read as `condition * payload`: a product of two values, either way
/// round, or `term` itself as a condition with an implicit payload of 1.
fn selector_terms(dfg: &DataFlowGraph, term: ValueId) -> Vec<(ValueId, Option<ValueId>)> {
    let mut terms = vec![(term, None)];
    if let Value::Instruction { instruction, .. } = dfg[term]
        && let Instruction::Binary(Binary { lhs, operator: BinaryOp::Mul { .. }, rhs }) =
            dfg[instruction]
    {
        terms.push((lhs, Some(rhs)));
        terms.push((rhs, Some(lhs)));
    }
    terms
}

fn is_not_of(dfg: &DataFlowGraph, value: ValueId, of: ValueId) -> bool {
    matches!(dfg[value], Value::Instruction { instruction, .. }
        if dfg[instruction] == Instruction::Not(of))
}

/// Bound for `lhs + rhs` when it selects between two values: `c * a + not(c) * b` (as built when
/// merging the branches of an `if`), where `c` is 0 or 1, so at most one term is non-zero and the
/// sum needs no more bits than the wider of `a` and `b`. Returns `None` for any other sum.
fn selected_sum_bound(
    dfg: &DataFlowGraph,
    lhs: ValueId,
    rhs: ValueId,
    required: &mut impl FnMut(ValueId) -> u32,
) -> Option<u32> {
    for (lhs_condition, lhs_payload) in selector_terms(dfg, lhs) {
        for (rhs_condition, rhs_payload) in selector_terms(dfg, rhs) {
            let complementary = is_not_of(dfg, rhs_condition, lhs_condition)
                || is_not_of(dfg, lhs_condition, rhs_condition);
            if complementary && required(lhs_condition) == 1 && required(rhs_condition) == 1 {
                let lhs_bits = lhs_payload.map_or(1, &mut *required);
                let rhs_bits = rhs_payload.map_or(1, &mut *required);
                return Some(lhs_bits.max(rhs_bits));
            }
        }
    }
    None
}

/// Pre-check condition for [`Function::checked_to_unchecked`].
///
/// Panics if:
///   - The function contains any checked signed binary operations (add, sub, mul).
///   - These should have already been converted by the `expand_signed_checks` pass.
#[cfg(debug_assertions)]
fn checked_to_unchecked_pre_check(func: &Function) {
    // expand_signed_checks must have run
    super::checks::for_each_instruction(func, |instruction, dfg| {
        super::checks::assert_not_checked_signed_add_sub_mul(instruction, dfg);
    });
}

#[cfg(test)]
mod tests {
    use crate::{assert_ssa_snapshot, ssa::ssa_gen::Ssa};

    #[test]
    fn checked_to_unchecked_when_casting_two_u16_to_u32_then_adding() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u16, v1: u16):
            v2 = cast v0 as u32
            v3 = cast v1 as u32
            v4 = add v2, v3
            return v4
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u16, v1: u16):
            v2 = cast v0 as u32
            v3 = cast v1 as u32
            v4 = unchecked_add v2, v3
            return v4
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_subtracting_u32() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u16):
            v1 = cast v0 as u32
            v2 = sub u32 65536, v1
            return v2
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u16):
            v1 = cast v0 as u32
            v3 = unchecked_sub u32 65536, v1
            return v3
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_subtracting_from_1_a_value_that_has_1_bit() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1):
            v1 = cast v0 as u32
            v3 = sub u32 1, v1
            return v3
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1):
            v1 = cast v0 as u32
            v3 = unchecked_sub u32 1, v1
            return v3
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_subtracting_from_255_a_value_that_has_8_bits() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8):
            v1 = cast v0 as u32
            v3 = sub u32 255, v1
            return v3
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8):
            v1 = cast v0 as u32
            v3 = unchecked_sub u32 255, v1
            return v3
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_multiplying_bools() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1):
            v2 = mul v0, v1
            return v2
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1):
            v2 = unchecked_mul v0, v1
            return v2
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_multiplying_upcasted_bool_with_u32() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u32):
            v2 = cast v0 as u32
            v3 = mul v2, v1
            return v2
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u32):
            v2 = cast v0 as u32
            v3 = unchecked_mul v2, v1
            return v2
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_multiplying_two_upcasted_bools_to_u32_then_multiplying_again() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u32):
            v3 = cast v0 as u32
            v4 = cast v1 as u32
            v5 = mul v3, v4
            v6 = mul v2, v5
            return v6
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u32):
            v3 = cast v0 as u32
            v4 = cast v1 as u32
            v5 = unchecked_mul v3, v4
            v6 = unchecked_mul v2, v5
            return v6
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_adding_two_u32_truncated_to_16_bits() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u32, v1: u32):
            v2 = truncate v0 to 16 bits, max_bit_size: 33
            v3 = truncate v1 to 16 bits, max_bit_size: 33
            v4 = add v2, v3
            return v4
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u32, v1: u32):
            v2 = truncate v0 to 16 bits, max_bit_size: 33
            v3 = truncate v1 to 16 bits, max_bit_size: 33
            v4 = unchecked_add v2, v3
            return v4
        }
        ");
    }

    #[test]
    fn does_not_uncheck_mul_by_cast_of_unchecked_u1_add_in_acir() {
        // In ACIR `unchecked_add u1 v0, v1` is the field value 2 when both are 1, so after the
        // cast `v4` is not a 0/1 value and the product can exceed `u8`.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = cast v3 as u8
            v5 = mul v4, v2
            return v5
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = cast v3 as u8
            v5 = mul v4, v2
            return v5
        }
        ");
    }

    #[test]
    fn unchecks_add_of_cast_of_unchecked_u8_add_using_its_widened_bound_in_acir() {
        // `v4` can be 510 in ACIR, a 9-bit value, so `v4 + v5` still fits in `u16`.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = cast v3 as u16
            v5 = cast v2 as u16
            v6 = add v4, v5
            return v6
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = cast v3 as u16
            v5 = cast v2 as u16
            v6 = unchecked_add v4, v5
            return v6
        }
        ");
    }

    #[test]
    fn does_not_uncheck_sub_from_constant_of_cast_of_unchecked_u1_sub_in_acir() {
        // An unchecked ACIR sub can underflow to a field-negative value of any width.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1):
            v2 = unchecked_sub v0, v1
            v3 = cast v2 as u32
            v5 = sub u32 1, v3
            return v5
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1):
            v2 = unchecked_sub v0, v1
            v3 = cast v2 as u32
            v5 = sub u32 1, v3
            return v5
        }
        ");
    }

    #[test]
    fn unchecks_mul_by_truncated_unchecked_u1_add_in_acir() {
        // A truncation brings the unchecked result back to one bit.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = truncate v3 to 1 bits, max_bit_size: 2
            v5 = cast v4 as u8
            v6 = mul v5, v2
            return v6
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = truncate v3 to 1 bits, max_bit_size: 2
            v5 = cast v4 as u8
            v6 = unchecked_mul v5, v2
            return v6
        }
        ");
    }

    #[test]
    fn unchecks_mul_by_cast_of_unchecked_u1_add_in_brillig() {
        // In Brillig unchecked arithmetic wraps to the type width.
        let src = "
        brillig(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = cast v3 as u8
            v5 = mul v4, v2
            return v5
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        brillig(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = cast v3 as u8
            v5 = unchecked_mul v4, v2
            return v5
        }
        ");
    }

    #[test]
    fn unchecks_mul_by_cast_of_merged_bools_in_acir() {
        // `v4 = v0 + not(v0) * v1` selects `v0` or `v1`, so it is still a single bit.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = not v0
            v4 = unchecked_mul v3, v1
            v5 = unchecked_add v0, v4
            v6 = cast v5 as u8
            v7 = mul v6, v2
            return v7
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = not v0
            v4 = unchecked_mul v3, v1
            v5 = unchecked_add v0, v4
            v6 = cast v5 as u8
            v7 = unchecked_mul v6, v2
            return v7
        }
        ");
    }

    #[test]
    fn does_not_treat_sum_as_selection_when_condition_is_not_a_bit_in_acir() {
        // `v3` can be 2, so `v4 = not v3` is field-negative and `v3 * v1 + v4 * v1` is not a
        // selection.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v0
            v4 = not v3
            v5 = unchecked_mul v3, v1
            v6 = unchecked_mul v4, v1
            v7 = unchecked_add v5, v6
            v8 = cast v7 as u8
            v9 = mul v8, v2
            return v9
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v0
            v4 = not v3
            v5 = unchecked_mul v3, v1
            v6 = unchecked_mul v4, v1
            v7 = unchecked_add v5, v6
            v8 = cast v7 as u8
            v9 = mul v8, v2
            return v9
        }
        ");
    }

    #[test]
    fn does_not_uncheck_mul_by_cast_of_not_of_unchecked_u1_add_in_acir() {
        // `v4 = not v3` is `1 - 2`, field-negative, when `v3` is 2.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = not v3
            v5 = cast v4 as u8
            v6 = mul v5, v2
            return v6
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1, v2: u8):
            v3 = unchecked_add v0, v1
            v4 = not v3
            v5 = cast v4 as u8
            v6 = mul v5, v2
            return v6
        }
        ");
    }

    #[test]
    fn long_chain_falls_back_to_the_field_width_in_acir() {
        // Past the depth limit the bound is the field width, so the add stays checked even though
        // every product in the chain is a single bit.
        let mut src = String::from("acir(inline) fn main f0 {\n  b0(v0: u1, v1: u8):\n");
        src.push_str("    v2 = unchecked_mul v0, v0\n");
        for i in 3..1203 {
            src.push_str(&format!("    v{i} = unchecked_mul v{}, v0\n", i - 1));
        }
        src.push_str("    v1203 = cast v1202 as u32\n    v1204 = cast v1 as u32\n");
        src.push_str("    v1205 = add v1203, v1204\n    return v1205\n}\n");
        let ssa = Ssa::from_str(&src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        let printed = ssa.to_string();
        assert!(printed.contains("v1205 = add v1203, v1204"), "{printed}");
    }

    #[test]
    fn does_not_uncheck_mul_of_unchecked_u8_add_by_a_bit_in_acir() {
        // `v3` can be 510 in ACIR, so multiplying it by a bit does not keep it within `u8`.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8, v2: u1):
            v3 = unchecked_add v0, v1
            v4 = cast v2 as u8
            v5 = mul v3, v4
            return v5
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8, v2: u1):
            v3 = unchecked_add v0, v1
            v4 = cast v2 as u8
            v5 = mul v3, v4
            return v5
        }
        ");
    }
}
