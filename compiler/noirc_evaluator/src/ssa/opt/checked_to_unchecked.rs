//! This SSA pass will turn checked unsigned binary additions, subtractions and multiplications
//! into unchecked ones if it's guaranteed that the operations cannot overflow.
//!
//! Signed checked binary operations should have already been converted to unchecked ones with
//! an explicit overflow check during [`super::expand_signed_checks`].

use rustc_hash::FxHashMap as HashMap;

use crate::ssa::{
    ir::{
        dfg::DataFlowGraph,
        function::Function,
        instruction::{Binary, BinaryOp, Instruction},
        types::{NumericType, max_unsigned_value_for_bit_size},
        value::ValueId,
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

        // Compute every value's unsigned bounds once. Converting an operation to unchecked only
        // happens when it cannot overflow, in which case neither the forward range nor the backward
        // range of any value changes, so this snapshot stays valid for the whole pass.
        let bounds = self.dfg.unsigned_value_bounds();

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
                    unsigned_operation_cannot_overflow(dfg, &bounds, lhs, rhs, |lhs, rhs| {
                        lhs.checked_add(rhs)
                    })
                }
                BinaryOp::Sub { unchecked: false } => {
                    let Some(&(lhs_min, _)) = bounds.get(&lhs) else {
                        return;
                    };
                    let Some(&(_, rhs_max)) = bounds.get(&rhs) else {
                        return;
                    };

                    lhs_min >= rhs_max
                }
                BinaryOp::Mul { unchecked: false } => {
                    unsigned_operation_cannot_overflow(dfg, &bounds, lhs, rhs, |lhs, rhs| {
                        lhs.checked_mul(rhs)
                    })
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

fn unsigned_operation_cannot_overflow(
    dfg: &DataFlowGraph,
    bounds: &HashMap<ValueId, (u128, u128)>,
    lhs: ValueId,
    rhs: ValueId,
    operation: impl FnOnce(u128, u128) -> Option<u128>,
) -> bool {
    // For unsigned monotonic operations, the maximum possible result is produced by the maximum
    // possible operands. If that result fits in the destination type, the checked op is redundant.
    let bit_size = dfg.type_of_value(lhs).bit_size();
    let Some(type_max) = max_unsigned_value_for_bit_size(bit_size) else {
        return false;
    };
    let Some(&(_, lhs_max)) = bounds.get(&lhs) else {
        return false;
    };
    let Some(&(_, rhs_max)) = bounds.get(&rhs) else {
        return false;
    };

    operation(lhs_max, rhs_max).is_some_and(|result| result <= type_max)
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
    fn checked_to_unchecked_uses_equality_constraint_ranges() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u128, v1: u128):
            v2 = add v0, v1
            constrain v2 == u128 100
            v3 = add v0, u128 1
            return v3
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u128, v1: u128):
            v2 = unchecked_add v0, v1
            constrain v2 == u128 100
            v5 = unchecked_add v0, u128 1
            return v5
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
    fn checked_to_unchecked_when_exact_add_range_fits() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v2 = mod v0, u8 9
            v3 = mod v1, u8 8
            v4 = add v2, v3
            return v4
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v3 = mod v0, u8 9
            v5 = mod v1, u8 8
            v6 = unchecked_add v3, v5
            return v6
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_exact_sub_range_cannot_underflow() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8):
            v1 = cast v0 as u32
            v2 = unchecked_add v1, u32 240
            v3 = sub u32 500, v2
            return v3
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8):
            v1 = cast v0 as u32
            v3 = unchecked_add v1, u32 240
            v5 = unchecked_sub u32 500, v3
            return v5
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_exact_mul_range_fits() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v2 = mod v0, u8 9
            v3 = mod v1, u8 8
            v4 = mul v2, v3
            return v4
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v3 = mod v0, u8 9
            v5 = mod v1, u8 8
            v6 = unchecked_mul v3, v5
            return v6
        }
        ");
    }

    #[test]
    fn checked_to_unchecked_when_result_range_check_prevents_overflow() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u128, v1: u128):
            v2 = add v0, v1
            range_check v2 to 8 bits
            return v2
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.checked_to_unchecked();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u128, v1: u128):
            v2 = unchecked_add v0, v1
            range_check v2 to 8 bits
            return v2
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
            v5 = mul v4, v2
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
            v7 = mul v6, v2
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
