//! This SSA pass removes `RangeCheck` instructions whose value the range analysis already proves
//! fits within the checked bit size.
//!
//! Instruction simplification only elides range checks that are redundant by type, since it runs on
//! every insertion and cannot afford the range analysis. The tighter, analysis-based elisions are
//! done here instead: once per function, so the range analysis is built a single time rather than
//! re-run for every instruction.

use crate::ssa::{
    ir::{function::Function, instruction::Instruction},
    ssa_gen::Ssa,
};

impl Ssa {
    /// See [`range_check_elision`][self] module for more information.
    #[tracing::instrument(level = "trace", skip(self))]
    pub(crate) fn remove_redundant_range_checks(mut self) -> Ssa {
        for function in self.functions.values_mut() {
            function.remove_redundant_range_checks();
        }
        self
    }
}

impl Function {
    fn remove_redundant_range_checks(&mut self) {
        if !self.runtime().is_acir() {
            return;
        }

        // A range check only constrains its own value; removing one never changes any value's
        // inferred bit width (the analysis does not use range checks on the unconstrained `bits`
        // path), so a single snapshot stays valid for the whole pass.
        let max_num_bits = self.dfg.value_max_num_bits();

        self.simple_optimization(|context| {
            let Instruction::RangeCheck { value, max_bit_size, .. } = context.instruction() else {
                return;
            };
            let value = *value;
            let max_bit_size = *max_bit_size;

            // `u32::MAX` keeps the check when the value is somehow absent from the snapshot.
            if max_num_bits.get(&value).copied().unwrap_or(u32::MAX) <= max_bit_size {
                context.remove_current_instruction();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_ssa_snapshot, ssa::ssa_gen::Ssa};

    #[test]
    fn keeps_range_check_on_unchecked_add_that_can_exceed_its_type() {
        // `v2` can be 510 in ACIR.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v2 = unchecked_add v0, v1
            range_check v2 to 8 bits
            return v2
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.remove_redundant_range_checks();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v2 = unchecked_add v0, v1
            return v2
        }
        ");
    }

    #[test]
    fn keeps_range_check_on_field_cast_of_unchecked_u1_add() {
        // `v3` can be 2 in ACIR.
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1):
            v2 = unchecked_add v0, v1
            v3 = cast v2 as Field
            range_check v3 to 1 bits
            return v3
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.remove_redundant_range_checks();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u1, v1: u1):
            v2 = unchecked_add v0, v1
            v3 = cast v2 as Field
            return v3
        }
        ");
    }

    #[test]
    fn removes_range_check_on_unchecked_add_that_fits_its_type() {
        let src = "
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v2 = truncate v0 to 7 bits, max_bit_size: 8
            v3 = truncate v1 to 7 bits, max_bit_size: 8
            v4 = unchecked_add v2, v3
            range_check v4 to 8 bits
            return v4
        }
        ";
        let ssa = Ssa::from_str(src).unwrap();
        let ssa = ssa.remove_redundant_range_checks();
        assert_ssa_snapshot!(ssa, @r"
        acir(inline) fn main f0 {
          b0(v0: u8, v1: u8):
            v2 = truncate v0 to 7 bits, max_bit_size: 8
            v3 = truncate v1 to 7 bits, max_bit_size: 8
            v4 = unchecked_add v2, v3
            return v4
        }
        ");
    }
}
