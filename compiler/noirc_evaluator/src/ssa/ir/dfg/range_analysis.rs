use acvm::AcirField;

use crate::ssa::ir::{
    instruction::Instruction,
    value::{Value, ValueId},
};

use super::DataFlowGraph;

/// Computes conservative numeric value ranges for SSA values.
pub(super) struct Analysis<'dfg> {
    dfg: &'dfg DataFlowGraph,
}

impl<'dfg> Analysis<'dfg> {
    pub(super) fn new(dfg: &'dfg DataFlowGraph) -> Self {
        Self { dfg }
    }

    /// Returns the maximum possible number of bits that `value` can potentially be.
    ///
    /// Should `value` be a numeric constant then this function will return the exact number of
    /// bits required, otherwise it will return the minimum number of bits based on type information.
    pub(super) fn bits(&self, value: ValueId) -> u32 {
        match self.dfg[value] {
            Value::Instruction { instruction, .. } => {
                let value_bit_size = self.dfg.type_of_value(value).bit_size();
                if let Instruction::Cast(original_value, _) = self.dfg[instruction] {
                    let original_bit_size = self.bits(original_value);
                    // We might have cast e.g. `u1` to `u8` to be able to do arithmetic,
                    // in which case we want to recover the original smaller bit size;
                    // OTOH if we cast down, then we don't need the higher original size.
                    value_bit_size.min(original_bit_size)
                } else {
                    value_bit_size
                }
            }

            Value::NumericConstant { constant, .. } => constant.num_bits(),
            _ => self.dfg.type_of_value(value).bit_size(),
        }
    }
}
