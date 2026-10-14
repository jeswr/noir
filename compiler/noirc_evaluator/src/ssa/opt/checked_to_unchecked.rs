//! This SSA pass will turn checked unsigned binary additions, subtractions and multiplications
//! into unchecked ones if it's guaranteed that the operations cannot overflow.
//!
//! Signed checked binary operations should have already been converted to unchecked ones with
//! an explicit overflow check during [`super::expand_signed_checks`].
use acvm::AcirField as _;
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
                        let max_rhs =
                            if max_rhs_bits == 128 { u128::MAX } else { (1 << max_rhs_bits) - 1 };

                        // `lhs` is a fixed constant and `rhs` is restricted such that `lhs - rhs >= 0`.
                        // For example: `lhs` is 1 and `rhs` max bitsize is 1, so at most it's `1 - 1`.
                        // Another example: `lhs` is 255 and `rhs` max bitsize is 8, so at most it's `255 - 255`.
                        lhs_const >= max_rhs.into()
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
                    // is at most 1, so `x * 0 = 0` or `x * 1 = x` — neither can overflow.
                    // This is sound as long as `required_bit_size` never returns 1 for a
                    // value that could actually exceed 1.
                    max_lhs_bits + max_rhs_bits <= bit_size
                        || max_lhs_bits == 1
                        || max_rhs_bits == 1
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
    if let Some(bits) = value_max_num_bits.get(&value) {
        return *bits;
    }

    let value_bit_size = dfg.type_of_value(value).bit_size();

    let bits = match dfg[value] {
        Value::Instruction { instruction, .. } => {
            match dfg[instruction] {
                Instruction::Cast(original_value, _) => {
                    let original_bit_size =
                        required_bit_size(dfg, original_value, value_max_num_bits);
                    // We might have cast e.g. `u1` to `u8` to be able to do arithmetic,
                    // in which case we want to recover the original smaller bit size;
                    // OTOH if we cast down, then we don't need the higher original size.
                    value_bit_size.min(original_bit_size)
                }
                Instruction::Binary(Binary { lhs, operator: BinaryOp::Mul { .. }, rhs })
                    if required_bit_size(dfg, lhs, value_max_num_bits) == 1
                        && required_bit_size(dfg, rhs, value_max_num_bits) == 1 =>
                {
                    // When multiplying two values, if their bitsize is 1 then the result's bitsize will be 1 too
                    1
                }
                Instruction::Truncate { value, bit_size, .. } => {
                    let value_bit_size =
                        value_bit_size.min(required_bit_size(dfg, value, value_max_num_bits));
                    value_bit_size.min(bit_size)
                }
                _ => value_bit_size,
            }
        }
        Value::NumericConstant { constant, .. } => constant.num_bits(),
        _ => value_bit_size,
    };

    assert!(bits <= value_bit_size);
    value_max_num_bits.insert(value, bits);

    bits
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
            v5 = unchecked_mul v4, v2
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
            v5 = unchecked_sub u32 1, v3
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
            v9 = unchecked_mul v8, v2
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
            v6 = unchecked_mul v5, v2
            return v6
        }
        ");
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
            v5 = unchecked_mul v3, v4
            return v5
        }
        ");
    }
}
