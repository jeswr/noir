use acvm::AcirField;
use rustc_hash::FxHashMap as HashMap;

use crate::ssa::ir::{
    instruction::{
        Binary, BinaryOp, Instruction, binary::try_convert_field_element_to_signed_integer,
    },
    types::{NumericType, Type, max_unsigned_value_for_bit_size},
    value::{Value, ValueId},
};

use super::DataFlowGraph;

fn ceil_div(numerator: u128, denominator: u128) -> u128 {
    debug_assert!(denominator > 0);

    let quotient = numerator / denominator;
    quotient + u128::from(numerator % denominator != 0)
}

fn signed_min_value(bit_size: u32) -> Option<i128> {
    match bit_size {
        1..=127 => Some(-(1i128 << (bit_size - 1))),
        128 => Some(i128::MIN),
        _ => None,
    }
}

fn signed_max_value(bit_size: u32) -> Option<i128> {
    match bit_size {
        1..=127 => Some((1i128 << (bit_size - 1)) - 1),
        128 => Some(i128::MAX),
        _ => None,
    }
}

fn sign_bit(bit_size: u32) -> Option<u128> {
    match bit_size {
        1..=128 => Some(1u128 << (bit_size - 1)),
        _ => None,
    }
}

fn signed_constant_value(constant: acvm::FieldElement, bit_size: u32) -> Option<i128> {
    if bit_size == 128 {
        constant.try_into_i128().or_else(|| constant.try_into_u128().map(|value| value as i128))
    } else {
        try_convert_field_element_to_signed_integer(constant, bit_size)
    }
}

fn signed_to_twos_complement(value: i128, bit_size: u32) -> Option<u128> {
    if value >= 0 {
        return u128::try_from(value).ok();
    }

    if bit_size == 128 {
        Some(value as u128)
    } else {
        Some((1u128 << bit_size) - value.unsigned_abs())
    }
}

fn unsigned_to_signed(value: u128, bit_size: u32) -> Option<i128> {
    if bit_size == 128 {
        Some(value as i128)
    } else if value < sign_bit(bit_size)? {
        i128::try_from(value).ok()
    } else {
        let magnitude = (1u128 << bit_size) - value;
        i128::try_from(magnitude).ok().map(|value| -value)
    }
}

fn fixed_nonnegative_shift(range: SignedRange, bit_size: u32) -> Option<u32> {
    if range.min == range.max && range.min >= 0 && range.max < i128::from(bit_size) {
        u32::try_from(range.max).ok()
    } else {
        None
    }
}

/// Computes conservative numeric value ranges for SSA values.
pub(super) struct Analysis<'dfg> {
    dfg: &'dfg DataFlowGraph,
}

#[derive(Clone, Copy)]
enum RangeSource<'facts> {
    Recursive,
    Facts(&'facts Facts),
}

impl<'facts> RangeSource<'facts> {
    fn range(self, analysis: &Analysis<'_>, value: ValueId) -> Option<ValueRange> {
        match self {
            Self::Recursive => analysis.range(value),
            Self::Facts(facts) => facts.range(value),
        }
    }

    /// Use fallback ranges only during recursive analysis.
    ///
    /// Fixed-point propagation must not invent facts for missing operands.
    fn range_or_fallback(
        self,
        analysis: &Analysis<'_>,
        value: ValueId,
        fallback: ValueRange,
    ) -> Option<ValueRange> {
        match self {
            Self::Recursive => Some(analysis.range(value).unwrap_or(fallback)),
            Self::Facts(facts) => facts.range(value),
        }
    }
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
        if let Some(range) = self.range(value) {
            let value_bit_size = self.dfg.type_of_value(value).bit_size();
            return value_bit_size.min(range.max_bits(value_bit_size));
        }

        match self.dfg[value] {
            Value::Instruction { instruction, .. } => {
                let value_bit_size = self.dfg.type_of_value(value).bit_size();
                match &self.dfg[instruction] {
                    // We might have cast e.g. `u1` to `u8` to be able to do arithmetic,
                    // in which case we want to recover the original smaller bit size;
                    // OTOH if we cast down, then we don't need the higher original size.
                    Instruction::Cast(original_value, _) => {
                        value_bit_size.min(self.bits(*original_value))
                    }
                    Instruction::Truncate { bit_size, .. } => value_bit_size.min(*bit_size),
                    _ => value_bit_size,
                }
            }
            Value::NumericConstant { constant, .. } => constant.num_bits(),
            _ => self.dfg.type_of_value(value).bit_size(),
        }
    }

    pub(super) fn constrained_bits(&self, value: ValueId) -> u32 {
        if let Some(range) = self.constrained_range(value) {
            let value_bit_size = self.dfg.type_of_value(value).bit_size();
            return value_bit_size.min(range.max_bits(value_bit_size));
        }

        self.bits(value)
    }

    pub(super) fn bounds(&self, value: ValueId) -> Option<(u128, u128)> {
        self.constrained_range(value)
            .and_then(ValueRange::into_unsigned)
            .map(|range| (range.min, range.max))
    }

    fn constrained_range(&self, value: ValueId) -> Option<ValueRange> {
        let mut facts = Facts::default();

        for (value, _) in self.dfg.values_iter() {
            if let Some(range) = self.range(value) {
                facts.set(value, range);
            }
        }

        // Branch-local or predicated constraints cannot be used as global value bounds.
        if self.dfg.blocks.len() == 1
            && !self.dfg.instructions.iter().any(|(_, instruction)| {
                matches!(instruction, Instruction::EnableSideEffectsIf { .. })
            })
        {
            for (_, instruction) in self.dfg.instructions.iter() {
                if let Instruction::RangeCheck { value, max_bit_size, .. } = instruction
                    && let Some(range) = self.range_check_range(*value, *max_bit_size)
                {
                    facts.refine(self.dfg, *value, range);
                }
            }

            // Propagate range information in both directions until the facts stop changing.
            // Safety bound; this normally exits early once no facts change.
            for _ in 0..=self.dfg.instructions.len() {
                let mut changed = false;

                for (instruction, instruction_data) in self.dfg.instructions.iter() {
                    let result = self
                        .dfg
                        .results
                        .get(&instruction)
                        .and_then(|results| results.first())
                        .copied();

                    if let Some(result) = result
                        && let Some(range) = self.instruction_range(
                            instruction_data,
                            result,
                            RangeSource::Facts(&facts),
                        )
                    {
                        changed |= facts.refine(self.dfg, result, range);
                    }

                    changed |= self.backward(instruction_data, result, &mut facts);
                    if let Instruction::Constrain(lhs, rhs, _) = instruction_data {
                        changed |= match (facts.range(*lhs), facts.range(*rhs)) {
                            (Some(lhs_range), Some(rhs_range)) => {
                                lhs_range.intersect(rhs_range).is_some_and(|range| {
                                    // Keep bitwise OR so both sides are refined.
                                    facts.refine(self.dfg, *lhs, range)
                                        | facts.refine(self.dfg, *rhs, range)
                                })
                            }
                            (Some(range), None) => facts.refine(self.dfg, *rhs, range),
                            (None, Some(range)) => facts.refine(self.dfg, *lhs, range),
                            (None, None) => false,
                        };
                    }
                }

                if !changed {
                    break;
                }
            }
        }

        facts.range(value)
    }

    /// Compute an instruction result range from either recursive analysis or known facts.
    fn instruction_range(
        &self,
        instruction: &Instruction,
        result: ValueId,
        source: RangeSource<'_>,
    ) -> Option<ValueRange> {
        let result_type = self.dfg.type_of_value(result);
        let value_bit_size = result_type.bit_size();

        match instruction {
            Instruction::Cast(original_value, _) => match result_type.as_ref() {
                Type::Numeric(NumericType::NativeField) => match source {
                    RangeSource::Recursive => None,
                    RangeSource::Facts(facts) => {
                        let original_type = self.dfg.type_of_value(*original_value);
                        facts
                            .range(*original_value)
                            .and_then(|range| range.cast_to_field(original_type.as_ref()))
                            .map(ValueRange::Unsigned)
                    }
                },
                Type::Numeric(NumericType::Unsigned { bit_size }) => {
                    let max = max_unsigned_value_for_bit_size(*bit_size)?;
                    let original_type = self.dfg.type_of_value(*original_value);
                    let original_range = source.range_or_fallback(
                        self,
                        *original_value,
                        ValueRange::Unsigned(Range::new(0, max)),
                    )?;
                    original_range
                        .cast_to_unsigned(original_type.as_ref(), *bit_size)
                        .map(ValueRange::Unsigned)
                }
                Type::Numeric(NumericType::Signed { bit_size }) => {
                    let original_type = self.dfg.type_of_value(*original_value);
                    let original_range = source.range_or_fallback(
                        self,
                        *original_value,
                        ValueRange::Signed(SignedRange::for_bit_size(*bit_size)?),
                    )?;
                    original_range
                        .cast_to_signed(original_type.as_ref(), *bit_size)
                        .map(ValueRange::Signed)
                }
                _ => None,
            },
            Instruction::Truncate { value: original_value, bit_size, .. } => self.truncate_range(
                *original_value,
                result_type.as_ref(),
                value_bit_size,
                *bit_size,
                source,
            ),
            Instruction::Binary(binary) => {
                let ranges = self.binary_ranges(binary, value_bit_size, source);
                binary.operator.forward(ranges)
            }
            Instruction::Not(original_value) => {
                self.not_range(*original_value, result_type.as_ref(), value_bit_size, source)
            }
            _ => None,
        }
    }

    /// Use a known result range to tighten operand ranges where the operation is invertible enough
    /// to produce sound bounds.
    fn backward(
        &self,
        instruction: &Instruction,
        result: Option<ValueId>,
        facts: &mut Facts,
    ) -> bool {
        let Some(result) = result else {
            return false;
        };
        let Some(result_range) = facts.range(result) else {
            return false;
        };

        match instruction {
            Instruction::Cast(original_value, _) => {
                let original_type = self.dfg.type_of_value(*original_value);
                let result_type = self.dfg.type_of_value(result);
                let is_lossless_cast = match (original_type.as_ref(), result_type.as_ref()) {
                    (
                        Type::Numeric(NumericType::Unsigned { .. }),
                        Type::Numeric(NumericType::NativeField),
                    ) => true,
                    (
                        Type::Numeric(NumericType::Unsigned { bit_size: original_bit_size }),
                        Type::Numeric(NumericType::Unsigned { bit_size: result_bit_size }),
                    ) => original_bit_size <= result_bit_size,
                    (
                        Type::Numeric(NumericType::Signed { bit_size: original_bit_size }),
                        Type::Numeric(NumericType::Signed { bit_size: result_bit_size }),
                    ) => original_bit_size <= result_bit_size,
                    _ => false,
                };
                if !is_lossless_cast {
                    return false;
                }
                facts.refine(self.dfg, *original_value, result_range)
            }
            Instruction::Binary(binary) => {
                let value_bit_size = self.dfg.type_of_value(binary.lhs).bit_size();
                let Some(ranges) =
                    self.binary_ranges(binary, value_bit_size, RangeSource::Facts(facts))
                else {
                    return false;
                };

                binary.operator.backward(BinaryBack::new(
                    self.dfg,
                    facts,
                    binary.lhs,
                    binary.rhs,
                    result_range,
                    ranges,
                ))
            }
            Instruction::Not(original_value) => {
                let original_type = self.dfg.type_of_value(*original_value);
                let Some(range) = result_range.not(original_type.as_ref()) else {
                    return false;
                };
                facts.refine(self.dfg, *original_value, range)
            }
            _ => false,
        }
    }

    fn range(&self, value: ValueId) -> Option<ValueRange> {
        let value_type = self.dfg.type_of_value(value);
        if !matches!(value_type.as_ref(), Type::Numeric(_)) {
            return None;
        }

        match self.dfg[value] {
            Value::NumericConstant { constant, typ } => self.constant_range(constant, typ),
            Value::Instruction { instruction, .. } => {
                let instruction = &self.dfg[instruction];
                match instruction {
                    Instruction::Cast(..)
                    | Instruction::Truncate { .. }
                    | Instruction::Binary(_)
                    | Instruction::Not(_) => {
                        self.instruction_range(instruction, value, RangeSource::Recursive)
                    }
                    _ => self.type_range(value),
                }
            }
            _ => self.type_range(value),
        }
    }

    fn binary_ranges(
        &self,
        binary: &Binary,
        value_bit_size: u32,
        source: RangeSource<'_>,
    ) -> Option<BinaryRanges> {
        let typ = self.dfg.type_of_value(binary.lhs).unwrap_numeric();
        let lhs = source.range(self, binary.lhs)?;
        let rhs = source.range(self, binary.rhs)?;
        match typ {
            NumericType::Unsigned { .. } => {
                BinaryRanges::unsigned(value_bit_size, lhs.into_unsigned()?, rhs.into_unsigned()?)
            }
            NumericType::Signed { .. } => {
                BinaryRanges::signed(value_bit_size, lhs.into_signed()?, rhs.into_signed()?)
            }
            NumericType::NativeField => None,
        }
    }

    fn type_range(&self, value: ValueId) -> Option<ValueRange> {
        let typ = self.dfg.type_of_value(value);
        let Type::Numeric(numeric_type) = typ.as_ref() else {
            return None;
        };
        ValueRange::for_type(numeric_type)
    }

    fn range_check_range(&self, value: ValueId, max_bit_size: u32) -> Option<ValueRange> {
        let max = max_unsigned_value_for_bit_size(max_bit_size)?;
        match self.dfg.type_of_value(value).as_ref() {
            Type::Numeric(NumericType::Unsigned { .. } | NumericType::NativeField) => {
                Some(ValueRange::Unsigned(Range::new(0, max)))
            }
            Type::Numeric(NumericType::Signed { bit_size }) if max_bit_size < *bit_size => {
                let max = i128::try_from(max).ok()?;
                Some(ValueRange::Signed(SignedRange::new(0, max)))
            }
            Type::Numeric(NumericType::Signed { bit_size }) => {
                Some(ValueRange::Signed(SignedRange::for_bit_size(*bit_size)?))
            }
            _ => None,
        }
    }

    fn constant_range(&self, constant: acvm::FieldElement, typ: NumericType) -> Option<ValueRange> {
        match typ {
            NumericType::Unsigned { .. } | NumericType::NativeField => {
                constant.try_into_u128().map(|value| ValueRange::Unsigned(Range::new(value, value)))
            }
            NumericType::Signed { bit_size } => signed_constant_value(constant, bit_size)
                .map(|value| ValueRange::Signed(SignedRange::new(value, value))),
        }
    }

    fn truncate_range(
        &self,
        original_value: ValueId,
        result_type: &Type,
        value_bit_size: u32,
        bit_size: u32,
        source: RangeSource<'_>,
    ) -> Option<ValueRange> {
        match result_type {
            Type::Numeric(NumericType::Unsigned { .. } | NumericType::NativeField) => {
                let max = max_unsigned_value_for_bit_size(value_bit_size.min(bit_size))?;
                let original_range = source.range_or_fallback(
                    self,
                    original_value,
                    ValueRange::Unsigned(Range::new(0, max)),
                )?;
                original_range
                    .cast_to_unsigned(result_type, value_bit_size.min(bit_size))
                    .map(|range| ValueRange::Unsigned(range.truncate_to(max)))
            }
            Type::Numeric(NumericType::Signed { bit_size: result_bit_size }) => {
                if bit_size >= *result_bit_size {
                    return source.range(self, original_value);
                }
                let max = i128::try_from(max_unsigned_value_for_bit_size(bit_size)?).ok()?;
                Some(ValueRange::Signed(SignedRange::new(0, max)))
            }
            _ => None,
        }
    }

    fn not_range(
        &self,
        original_value: ValueId,
        result_type: &Type,
        value_bit_size: u32,
        source: RangeSource<'_>,
    ) -> Option<ValueRange> {
        match result_type {
            Type::Numeric(NumericType::Unsigned { .. }) => {
                let type_max = max_unsigned_value_for_bit_size(value_bit_size)?;
                let original_range = source.range_or_fallback(
                    self,
                    original_value,
                    ValueRange::Unsigned(Range::new(0, type_max)),
                )?;
                Some(ValueRange::Unsigned(original_range.into_unsigned()?.not(type_max)))
            }
            Type::Numeric(NumericType::Signed { bit_size }) => {
                let original_range = source.range_or_fallback(
                    self,
                    original_value,
                    ValueRange::Signed(SignedRange::for_bit_size(*bit_size)?),
                )?;
                Some(ValueRange::Signed(original_range.into_signed()?.not()))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueRange {
    Unsigned(Range),
    Signed(SignedRange),
}

impl ValueRange {
    fn for_type(typ: &NumericType) -> Option<Self> {
        match typ {
            NumericType::Unsigned { bit_size } => {
                Some(Self::Unsigned(Range::new(0, max_unsigned_value_for_bit_size(*bit_size)?)))
            }
            NumericType::Signed { bit_size } => {
                Some(Self::Signed(SignedRange::for_bit_size(*bit_size)?))
            }
            NumericType::NativeField => None,
        }
    }

    fn max_bits(self, type_bit_size: u32) -> u32 {
        match self {
            Self::Unsigned(range) => range.max_bits(),
            Self::Signed(range) => range.max_bits(type_bit_size),
        }
    }

    fn intersect(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::Unsigned(lhs), Self::Unsigned(rhs)) => lhs.intersect(rhs).map(Self::Unsigned),
            (Self::Signed(lhs), Self::Signed(rhs)) => lhs.intersect(rhs).map(Self::Signed),
            _ => None,
        }
    }

    fn into_unsigned(self) -> Option<Range> {
        match self {
            Self::Unsigned(range) => Some(range),
            Self::Signed(_) => None,
        }
    }

    fn into_signed(self) -> Option<SignedRange> {
        match self {
            Self::Signed(range) => Some(range),
            Self::Unsigned(_) => None,
        }
    }

    fn clamp_to_type(self, typ: &NumericType) -> Option<Self> {
        match (self, typ) {
            (Self::Unsigned(range), NumericType::Unsigned { bit_size }) => {
                let type_max = max_unsigned_value_for_bit_size(*bit_size)?;
                Some(Self::Unsigned(Range::new(range.min.min(type_max), range.max.min(type_max))))
            }
            (Self::Unsigned(range), NumericType::NativeField) => Some(Self::Unsigned(range)),
            (Self::Signed(range), NumericType::Signed { bit_size }) => {
                Some(Self::Signed(range.intersect(SignedRange::for_bit_size(*bit_size)?)?))
            }
            _ => None,
        }
    }

    fn cast_to_field(self, source_type: &Type) -> Option<Range> {
        match self {
            Self::Unsigned(range) => Some(range),
            Self::Signed(range) => {
                let Type::Numeric(NumericType::Signed { bit_size }) = source_type else {
                    return None;
                };
                range.to_unsigned(*bit_size, *bit_size)
            }
        }
    }

    fn cast_to_unsigned(self, source_type: &Type, target_bit_size: u32) -> Option<Range> {
        match self {
            Self::Unsigned(range) => {
                let target_max = max_unsigned_value_for_bit_size(target_bit_size)?;
                Some(range.truncate_to(target_max))
            }
            Self::Signed(range) => {
                let Type::Numeric(NumericType::Signed { bit_size }) = source_type else {
                    return None;
                };
                range.to_unsigned(*bit_size, target_bit_size)
            }
        }
    }

    fn cast_to_signed(self, source_type: &Type, target_bit_size: u32) -> Option<SignedRange> {
        match self {
            Self::Signed(range) => {
                if range.fits_in_bits(target_bit_size) {
                    Some(range)
                } else {
                    SignedRange::for_bit_size(target_bit_size)
                }
            }
            Self::Unsigned(range) => {
                let Type::Numeric(NumericType::Unsigned { .. } | NumericType::NativeField) =
                    source_type
                else {
                    return None;
                };
                SignedRange::from_unsigned(range, target_bit_size)
            }
        }
    }

    fn not(self, typ: &Type) -> Option<Self> {
        match (self, typ) {
            (Self::Unsigned(range), Type::Numeric(NumericType::Unsigned { bit_size })) => {
                Some(Self::Unsigned(range.not(max_unsigned_value_for_bit_size(*bit_size)?)))
            }
            (Self::Signed(range), Type::Numeric(NumericType::Signed { .. })) => {
                Some(Self::Signed(range.not()))
            }
            _ => None,
        }
    }
}

#[derive(Default, Debug)]
struct Facts {
    ranges: HashMap<ValueId, ValueRange>,
}

impl Facts {
    fn range(&self, value: ValueId) -> Option<ValueRange> {
        self.ranges.get(&value).copied()
    }

    fn set(&mut self, value: ValueId, range: ValueRange) {
        self.ranges.insert(value, range);
    }

    /// Intersect `value`'s current range with `range`.
    ///
    /// Empty refinements are ignored. They can appear when independent conservative facts cannot
    /// overlap, and inventing a replacement singleton would make later inferences unsound.
    fn refine(&mut self, dfg: &DataFlowGraph, value: ValueId, range: ValueRange) -> bool {
        let value_type = dfg.type_of_value(value);
        let Type::Numeric(numeric_type) = value_type.as_ref() else {
            return false;
        };
        let Some(range) = range.clamp_to_type(numeric_type) else {
            return false;
        };

        let Some(existing) = self.range(value) else {
            self.set(value, range);
            return true;
        };

        let Some(range) = existing.intersect(range) else {
            return false;
        };

        if range != existing {
            self.set(value, range);
            true
        } else {
            false
        }
    }
}

/// Inclusive range of possible values for an unsigned SSA value.
///
/// These ranges are deliberately conservative: if an operation can overflow or truncate, the
/// lower bound falls back to zero rather than assuming the overflowing case is unreachable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    min: u128,
    max: u128,
}

impl Range {
    fn new(min: u128, max: u128) -> Self {
        debug_assert!(min <= max);
        Self { min, max }
    }

    fn intersect(self, other: Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min <= max { Some(Self::new(min, max)) } else { None }
    }

    fn truncate_to(self, max: u128) -> Self {
        if self.max <= max {
            self
        } else {
            // If the source can exceed the target max, truncation may wrap to any target value.
            Self::new(0, max)
        }
    }

    fn not(self, type_max: u128) -> Self {
        Self::new(type_max - self.max, type_max - self.min)
    }

    fn increasing_result(
        self,
        rhs: Self,
        type_max: u128,
        operation: impl Fn(u128, u128) -> Option<u128>,
    ) -> Self {
        let max_result = operation(self.max, rhs.max);
        let may_exceed_type = max_result.is_none_or(|result| result > type_max);
        let max = max_result.unwrap_or(type_max).min(type_max);
        let min = if may_exceed_type {
            0
        } else {
            operation(self.min, rhs.min).filter(|result| *result <= type_max).unwrap_or(0)
        };
        Self::new(min, max)
    }

    fn max_bits(self) -> u32 {
        u128::BITS - self.max.leading_zeros()
    }
}

/// Inclusive range of possible logical values for a signed SSA value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SignedRange {
    min: i128,
    max: i128,
}

impl SignedRange {
    fn new(min: i128, max: i128) -> Self {
        debug_assert!(min <= max);
        Self { min, max }
    }

    fn for_bit_size(bit_size: u32) -> Option<Self> {
        Some(Self::new(signed_min_value(bit_size)?, signed_max_value(bit_size)?))
    }

    fn intersect(self, other: Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min <= max { Some(Self::new(min, max)) } else { None }
    }

    fn fits_in_bits(self, bit_size: u32) -> bool {
        let Some(type_range) = Self::for_bit_size(bit_size) else {
            return false;
        };
        type_range.min <= self.min && self.max <= type_range.max
    }

    fn max_bits(self, type_bit_size: u32) -> u32 {
        if self.min < 0 { type_bit_size } else { u128::BITS - (self.max as u128).leading_zeros() }
    }

    fn not(self) -> Self {
        Self::new(!self.max, !self.min)
    }

    fn add(self, rhs: Self, type_range: Self) -> Self {
        self.checked_result(rhs, type_range, i128::checked_add)
    }

    fn sub(self, rhs: Self, type_range: Self) -> Self {
        let candidates = [self.min.checked_sub(rhs.max), self.max.checked_sub(rhs.min)];
        Self::from_checked_candidates(candidates, type_range)
    }

    fn mul(self, rhs: Self, type_range: Self) -> Self {
        self.checked_result(rhs, type_range, i128::checked_mul)
    }

    fn div(self, rhs: Self, type_range: Self) -> Self {
        if rhs.contains(0) || (self.contains(type_range.min) && rhs.contains(-1)) {
            return type_range;
        }

        self.checked_result(rhs, type_range, i128::checked_div)
    }

    fn modulo(self, rhs: Self, type_range: Self) -> Self {
        if rhs.contains(0) || (self.contains(type_range.min) && rhs.contains(-1)) {
            return type_range;
        }

        let max_abs_rhs = rhs.max_abs().saturating_sub(1);
        let max_magnitude = max_abs_rhs.min(i128::MAX as u128) as i128;

        if self.max < 0 {
            let lhs_magnitude = self.max_abs().min(i128::MAX as u128) as i128;
            Self::new(-max_magnitude.min(lhs_magnitude), 0)
        } else if self.min >= 0 {
            Self::new(0, max_magnitude.min(self.max))
        } else {
            Self::new(-max_magnitude, max_magnitude)
        }
    }

    fn shl(self, rhs: Self, bit_size: u32, type_range: Self) -> Self {
        let Some(shift) = fixed_nonnegative_shift(rhs, bit_size) else {
            return type_range;
        };

        self.checked_result(Self::new(shift.into(), shift.into()), type_range, |lhs, rhs| {
            lhs.checked_shl(u32::try_from(rhs).ok()?)
        })
    }

    fn shr(self, rhs: Self, bit_size: u32, type_range: Self) -> Self {
        let Some(shift) = fixed_nonnegative_shift(rhs, bit_size) else {
            return type_range;
        };

        Self::new(self.min >> shift, self.max >> shift)
    }

    fn checked_result(
        self,
        rhs: Self,
        type_range: Self,
        operation: impl Fn(i128, i128) -> Option<i128>,
    ) -> Self {
        let candidates = [
            operation(self.min, rhs.min),
            operation(self.min, rhs.max),
            operation(self.max, rhs.min),
            operation(self.max, rhs.max),
        ];
        Self::from_checked_candidates(candidates, type_range)
    }

    fn from_checked_candidates<const N: usize>(
        candidates: [Option<i128>; N],
        type_range: Self,
    ) -> Self {
        let mut min = i128::MAX;
        let mut max = i128::MIN;
        for candidate in candidates {
            let Some(candidate) = candidate else {
                return type_range;
            };
            if candidate < type_range.min || candidate > type_range.max {
                return type_range;
            }
            min = min.min(candidate);
            max = max.max(candidate);
        }
        Self::new(min, max)
    }

    fn contains(self, value: i128) -> bool {
        self.min <= value && value <= self.max
    }

    fn max_abs(self) -> u128 {
        self.min.unsigned_abs().max(self.max.unsigned_abs())
    }

    fn to_unsigned(self, source_bit_size: u32, target_bit_size: u32) -> Option<Range> {
        let target_max = max_unsigned_value_for_bit_size(target_bit_size)?;
        if self.min >= 0 {
            return Some(Range::new(self.min as u128, self.max as u128).truncate_to(target_max));
        }

        if self.max >= 0 || target_bit_size < source_bit_size {
            return Some(Range::new(0, target_max));
        }

        let min = signed_to_twos_complement(self.min, source_bit_size)?;
        let max = signed_to_twos_complement(self.max, source_bit_size)?;
        Some(Range::new(min, max).truncate_to(target_max))
    }

    fn from_unsigned(range: Range, target_bit_size: u32) -> Option<Self> {
        let type_range = Self::for_bit_size(target_bit_size)?;
        let sign_bit = sign_bit(target_bit_size)?;
        let type_max = max_unsigned_value_for_bit_size(target_bit_size)?;

        if range.max <= type_range.max as u128 {
            return Some(Self::new(i128::try_from(range.min).ok()?, range.max as i128));
        }

        if range.min >= sign_bit && range.max <= type_max {
            let min = unsigned_to_signed(range.min, target_bit_size)?;
            let max = unsigned_to_signed(range.max, target_bit_size)?;
            return Some(Self::new(min, max));
        }

        Some(type_range)
    }
}

#[derive(Clone, Copy)]
enum BinaryRanges {
    Unsigned(UnsignedBinaryRanges),
    Signed(SignedBinaryRanges),
}

impl BinaryRanges {
    fn unsigned(bit_size: u32, lhs: Range, rhs: Range) -> Option<Self> {
        UnsignedBinaryRanges::new(bit_size, lhs, rhs).map(Self::Unsigned)
    }

    fn signed(bit_size: u32, lhs: SignedRange, rhs: SignedRange) -> Option<Self> {
        SignedBinaryRanges::new(bit_size, lhs, rhs).map(Self::Signed)
    }

    fn map_unsigned_signed(
        self,
        unsigned: impl FnOnce(UnsignedBinaryRanges) -> Range,
        signed: impl FnOnce(SignedBinaryRanges) -> SignedRange,
    ) -> ValueRange {
        match self {
            Self::Unsigned(ranges) => ValueRange::Unsigned(unsigned(ranges)),
            Self::Signed(ranges) => ValueRange::Signed(signed(ranges)),
        }
    }

    fn add(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::add, SignedBinaryRanges::add)
    }

    fn sub(self, unchecked: bool) -> ValueRange {
        self.map_unsigned_signed(|ranges| ranges.sub(unchecked), SignedBinaryRanges::sub)
    }

    fn mul(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::mul, SignedBinaryRanges::mul)
    }

    fn div(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::div, SignedBinaryRanges::div)
    }

    fn modulo(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::modulo, SignedBinaryRanges::modulo)
    }

    fn bitand(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::bitand, SignedBinaryRanges::bitwise)
    }

    fn bit_or_xor(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::bit_or_xor, SignedBinaryRanges::bitwise)
    }

    fn shl(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::shl, SignedBinaryRanges::shl)
    }

    fn shr(self) -> ValueRange {
        self.map_unsigned_signed(UnsignedBinaryRanges::shr, SignedBinaryRanges::shr)
    }
}

#[derive(Clone, Copy)]
struct UnsignedBinaryRanges {
    lhs: Range,
    rhs: Range,
    bit_size: u32,
    type_max: u128,
}

impl UnsignedBinaryRanges {
    fn new(bit_size: u32, lhs: Range, rhs: Range) -> Option<Self> {
        Some(Self { lhs, rhs, bit_size, type_max: max_unsigned_value_for_bit_size(bit_size)? })
    }

    fn add(self) -> Range {
        self.lhs.increasing_result(self.rhs, self.type_max, u128::checked_add)
    }

    fn sub(self, unchecked: bool) -> Range {
        if unchecked {
            if self.lhs.min >= self.rhs.max {
                Range::new(self.lhs.min - self.rhs.max, self.lhs.max - self.rhs.min)
            } else {
                Range::new(0, self.type_max)
            }
        } else {
            let min = self.lhs.min.checked_sub(self.rhs.max).unwrap_or(0);
            let max = self.lhs.max.saturating_sub(self.rhs.min);
            Range::new(min, max)
        }
    }

    fn mul(self) -> Range {
        self.lhs.increasing_result(self.rhs, self.type_max, u128::checked_mul)
    }

    fn div(self) -> Range {
        let max = if self.rhs.min == 0 { self.lhs.max } else { self.lhs.max / self.rhs.min };
        let min = if self.rhs.min == 0 { 0 } else { self.lhs.min / self.rhs.max };
        Range::new(min, max)
    }

    fn modulo(self) -> Range {
        let max = if self.rhs.min == 0 {
            self.lhs.max
        } else {
            self.lhs.max.min(self.rhs.max.saturating_sub(1))
        };
        Range::new(0, max)
    }

    fn bitand(self) -> Range {
        Range::new(0, self.lhs.max.min(self.rhs.max))
    }

    fn bit_or_xor(self) -> Range {
        let max_bits = self.lhs.max_bits().max(self.rhs.max_bits());
        let max =
            max_unsigned_value_for_bit_size(max_bits).unwrap_or(self.type_max).min(self.type_max);
        Range::new(0, max)
    }

    fn shl(self) -> Range {
        if self.rhs.min == self.rhs.max && self.rhs.max < 128 {
            let shift = self.rhs.max as u32;
            let max_shifted = self.lhs.max.checked_shl(shift);
            let overflow_possible = max_shifted.is_none_or(|shifted| shifted > self.type_max);
            let max = max_shifted.unwrap_or(self.type_max).min(self.type_max);
            let min = if overflow_possible {
                0
            } else {
                self.lhs
                    .min
                    .checked_shl(shift)
                    .filter(|shifted| *shifted <= self.type_max)
                    .unwrap_or(0)
            };
            Range::new(min, max)
        } else {
            Range::new(0, self.type_max)
        }
    }

    fn shr(self) -> Range {
        if self.rhs.max < u128::from(self.bit_size) {
            let max = self.lhs.max >> self.rhs.min as u32;
            let min = self.lhs.min >> self.rhs.max as u32;
            Range::new(min, max)
        } else {
            Range::new(0, self.type_max)
        }
    }
}

#[derive(Clone, Copy)]
struct SignedBinaryRanges {
    lhs: SignedRange,
    rhs: SignedRange,
    bit_size: u32,
    type_range: SignedRange,
}

impl SignedBinaryRanges {
    fn new(bit_size: u32, lhs: SignedRange, rhs: SignedRange) -> Option<Self> {
        Some(Self { lhs, rhs, bit_size, type_range: SignedRange::for_bit_size(bit_size)? })
    }

    fn add(self) -> SignedRange {
        self.lhs.add(self.rhs, self.type_range)
    }

    fn sub(self) -> SignedRange {
        self.lhs.sub(self.rhs, self.type_range)
    }

    fn mul(self) -> SignedRange {
        self.lhs.mul(self.rhs, self.type_range)
    }

    fn div(self) -> SignedRange {
        self.lhs.div(self.rhs, self.type_range)
    }

    fn modulo(self) -> SignedRange {
        self.lhs.modulo(self.rhs, self.type_range)
    }

    fn bitwise(self) -> SignedRange {
        self.type_range
    }

    fn shl(self) -> SignedRange {
        self.lhs.shl(self.rhs, self.bit_size, self.type_range)
    }

    fn shr(self) -> SignedRange {
        self.lhs.shr(self.rhs, self.bit_size, self.type_range)
    }
}

enum BinaryBack<'a> {
    Unsigned(UnsignedBinaryBack<'a>),
    Signed(SignedBinaryBack<'a>),
}

impl<'a> BinaryBack<'a> {
    fn new(
        dfg: &'a DataFlowGraph,
        facts: &'a mut Facts,
        lhs: ValueId,
        rhs: ValueId,
        result: ValueRange,
        ranges: BinaryRanges,
    ) -> Option<Self> {
        match (result, ranges) {
            (ValueRange::Unsigned(result), BinaryRanges::Unsigned(ranges)) => {
                Some(Self::Unsigned(UnsignedBinaryBack { dfg, facts, lhs, rhs, result, ranges }))
            }
            (ValueRange::Signed(result), BinaryRanges::Signed(ranges)) => {
                Some(Self::Signed(SignedBinaryBack { dfg, facts, lhs, rhs, result, ranges }))
            }
            _ => None,
        }
    }
}

struct UnsignedBinaryBack<'a> {
    dfg: &'a DataFlowGraph,
    facts: &'a mut Facts,
    lhs: ValueId,
    rhs: ValueId,
    result: Range,
    ranges: UnsignedBinaryRanges,
}

impl<'a> UnsignedBinaryBack<'a> {
    fn add(&self, unchecked: bool) -> Option<OperandRanges> {
        if unchecked && self.result_may_wrap(u128::checked_add) {
            return None;
        }

        let lhs_max = self.result.max.saturating_sub(self.ranges.rhs.min);
        let rhs_max = self.result.max.saturating_sub(self.ranges.lhs.min);
        let lhs_min = self.result.min.saturating_sub(self.ranges.rhs.max);
        let rhs_min = self.result.min.saturating_sub(self.ranges.lhs.max);

        Some(
            self.operands()
                .tighten_lhs_bounds(lhs_min, lhs_max)
                .tighten_rhs_bounds(rhs_min, rhs_max),
        )
    }

    fn sub(&self, unchecked: bool) -> Option<OperandRanges> {
        if unchecked && self.ranges.lhs.min < self.ranges.rhs.max {
            return None;
        }

        let lhs_min =
            self.result.min.checked_add(self.ranges.rhs.min).unwrap_or(self.ranges.type_max);
        let lhs_max =
            self.result.max.checked_add(self.ranges.rhs.max).unwrap_or(self.ranges.type_max);
        let rhs_min = self.ranges.lhs.min.saturating_sub(self.result.max);
        let rhs_max = self.ranges.lhs.max.saturating_sub(self.result.min);

        Some(
            self.operands()
                .tighten_lhs_bounds(lhs_min, lhs_max)
                .tighten_rhs_bounds(rhs_min, rhs_max),
        )
    }

    fn mul(&self, unchecked: bool) -> Option<OperandRanges> {
        if unchecked && self.result_may_wrap(u128::checked_mul) {
            return None;
        }

        let mut operands = self.operands();
        if self.ranges.rhs.min > 0 {
            operands = operands.tighten_lhs_bounds(
                ceil_div(self.result.min, self.ranges.rhs.max),
                self.result.max / self.ranges.rhs.min,
            );
        }
        if self.ranges.lhs.min > 0 {
            operands = operands.tighten_rhs_bounds(
                ceil_div(self.result.min, self.ranges.lhs.max),
                self.result.max / self.ranges.lhs.min,
            );
        }
        Some(operands)
    }

    fn div(&self) -> Option<OperandRanges> {
        let mut operands = self.operands();

        if self.ranges.rhs.max > 0 {
            let lhs_max = self
                .result
                .max
                .checked_add(1)
                .and_then(|max_plus_one| max_plus_one.checked_mul(self.ranges.rhs.max))
                .and_then(|exclusive_bound| exclusive_bound.checked_sub(1))
                .unwrap_or(self.ranges.type_max)
                .min(self.ranges.type_max);
            operands = operands.tighten_lhs_bounds(0, lhs_max);
        }

        if self.ranges.rhs.min > 0 {
            let lhs_min = self
                .result
                .min
                .checked_mul(self.ranges.rhs.min)
                .unwrap_or(self.ranges.type_max)
                .min(self.ranges.type_max);
            operands = operands.tighten_lhs_bounds(lhs_min, self.ranges.type_max);
        }

        Some(operands)
    }

    fn bitor(&self) -> Option<OperandRanges> {
        Some(
            self.operands()
                .tighten_lhs(Range::new(0, self.result.max))
                .tighten_rhs(Range::new(0, self.result.max)),
        )
    }

    fn shl(&self) -> Option<OperandRanges> {
        if self.ranges.rhs.min != self.ranges.rhs.max || self.ranges.rhs.max >= 128 {
            return None;
        }

        let shift = self.ranges.rhs.max as u32;
        if !self.ranges.lhs.max.checked_shl(shift).is_some_and(|max| max <= self.ranges.type_max) {
            return None;
        }

        Some(self.operands().tighten_lhs(Range::new(0, self.result.max >> shift)))
    }

    fn apply(self, operands: Option<OperandRanges>) -> bool {
        let Some(operands) = operands else {
            return false;
        };

        self.facts.refine(self.dfg, self.lhs, ValueRange::Unsigned(operands.lhs))
            | self.facts.refine(self.dfg, self.rhs, ValueRange::Unsigned(operands.rhs))
    }

    fn operands(&self) -> OperandRanges {
        OperandRanges { lhs: self.ranges.lhs, rhs: self.ranges.rhs }
    }

    fn result_may_wrap(&self, operation: impl FnOnce(u128, u128) -> Option<u128>) -> bool {
        operation(self.ranges.lhs.max, self.ranges.rhs.max)
            .is_none_or(|result| result > self.ranges.type_max)
    }
}

struct OperandRanges {
    lhs: Range,
    rhs: Range,
}

impl OperandRanges {
    // Empty intersections are ignored to match `Facts::refine`.
    fn tighten_lhs_bounds(self, min: u128, max: u128) -> Self {
        if min <= max { self.tighten_lhs(Range::new(min, max)) } else { self }
    }

    fn tighten_rhs_bounds(self, min: u128, max: u128) -> Self {
        if min <= max { self.tighten_rhs(Range::new(min, max)) } else { self }
    }

    fn tighten_lhs(mut self, range: Range) -> Self {
        if let Some(range) = self.lhs.intersect(range) {
            self.lhs = range;
        }
        self
    }

    fn tighten_rhs(mut self, range: Range) -> Self {
        if let Some(range) = self.rhs.intersect(range) {
            self.rhs = range;
        }
        self
    }
}

struct SignedBinaryBack<'a> {
    dfg: &'a DataFlowGraph,
    facts: &'a mut Facts,
    lhs: ValueId,
    rhs: ValueId,
    result: SignedRange,
    ranges: SignedBinaryRanges,
}

impl<'a> SignedBinaryBack<'a> {
    fn add(&self) -> Option<SignedOperandRanges> {
        let lhs = self.bounds(
            self.result.min.checked_sub(self.ranges.rhs.max),
            self.result.max.checked_sub(self.ranges.rhs.min),
        )?;
        let rhs = self.bounds(
            self.result.min.checked_sub(self.ranges.lhs.max),
            self.result.max.checked_sub(self.ranges.lhs.min),
        )?;
        Some(self.operands().tighten_lhs(lhs).tighten_rhs(rhs))
    }

    fn sub(&self) -> Option<SignedOperandRanges> {
        let lhs = self.bounds(
            self.result.min.checked_add(self.ranges.rhs.min),
            self.result.max.checked_add(self.ranges.rhs.max),
        )?;
        let rhs = self.bounds(
            self.ranges.lhs.min.checked_sub(self.result.max),
            self.ranges.lhs.max.checked_sub(self.result.min),
        )?;
        Some(self.operands().tighten_lhs(lhs).tighten_rhs(rhs))
    }

    fn apply(self, operands: Option<SignedOperandRanges>) -> bool {
        let Some(operands) = operands else {
            return false;
        };

        self.facts.refine(self.dfg, self.lhs, ValueRange::Signed(operands.lhs))
            | self.facts.refine(self.dfg, self.rhs, ValueRange::Signed(operands.rhs))
    }

    fn operands(&self) -> SignedOperandRanges {
        SignedOperandRanges { lhs: self.ranges.lhs, rhs: self.ranges.rhs }
    }

    fn bounds(&self, min: Option<i128>, max: Option<i128>) -> Option<SignedRange> {
        let min = min?.max(self.ranges.type_range.min);
        let max = max?.min(self.ranges.type_range.max);
        (min <= max).then(|| SignedRange::new(min, max))
    }
}

struct SignedOperandRanges {
    lhs: SignedRange,
    rhs: SignedRange,
}

impl SignedOperandRanges {
    fn tighten_lhs(mut self, range: SignedRange) -> Self {
        if let Some(range) = self.lhs.intersect(range) {
            self.lhs = range;
        }
        self
    }

    fn tighten_rhs(mut self, range: SignedRange) -> Self {
        if let Some(range) = self.rhs.intersect(range) {
            self.rhs = range;
        }
        self
    }
}

impl BinaryOp {
    fn forward(self, ranges: Option<BinaryRanges>) -> Option<ValueRange> {
        match self {
            BinaryOp::Eq | BinaryOp::Lt => Some(ValueRange::Unsigned(Range::new(0, 1))),
            BinaryOp::Add { .. } => Some(ranges?.add()),
            BinaryOp::Sub { unchecked } => Some(ranges?.sub(unchecked)),
            BinaryOp::Mul { .. } => Some(ranges?.mul()),
            BinaryOp::Div => Some(ranges?.div()),
            BinaryOp::Mod => Some(ranges?.modulo()),
            BinaryOp::And => Some(ranges?.bitand()),
            BinaryOp::Or | BinaryOp::Xor => Some(ranges?.bit_or_xor()),
            BinaryOp::Shl => Some(ranges?.shl()),
            BinaryOp::Shr => Some(ranges?.shr()),
        }
    }

    fn backward(self, back: Option<BinaryBack<'_>>) -> bool {
        match back {
            Some(BinaryBack::Unsigned(back)) => {
                let operands = match self {
                    BinaryOp::Add { unchecked } => back.add(unchecked),
                    BinaryOp::Sub { unchecked } => back.sub(unchecked),
                    BinaryOp::Mul { unchecked } => back.mul(unchecked),
                    BinaryOp::Div => back.div(),
                    BinaryOp::Or => back.bitor(),
                    BinaryOp::Shl => back.shl(),
                    BinaryOp::Mod
                    | BinaryOp::And
                    | BinaryOp::Xor
                    | BinaryOp::Shr
                    | BinaryOp::Eq
                    | BinaryOp::Lt => None,
                };
                back.apply(operands)
            }
            Some(BinaryBack::Signed(back)) => {
                let operands = match self {
                    BinaryOp::Add { unchecked: false } => back.add(),
                    BinaryOp::Sub { unchecked: false } => back.sub(),
                    _ => None,
                };
                back.apply(operands)
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const U8_BITS: u32 = 8;

    fn u8_ranges(lhs: Range, rhs: Range) -> UnsignedBinaryRanges {
        UnsignedBinaryRanges::new(U8_BITS, lhs, rhs).unwrap()
    }

    fn apply_u8_back(
        lhs_range: Range,
        rhs_range: Range,
        result: Range,
        operator: BinaryOp,
    ) -> (bool, Range, Range) {
        let ranges = u8_ranges(lhs_range, rhs_range);
        let mut dfg = DataFlowGraph::default();
        let block = dfg.make_block();
        let lhs = dfg.add_block_parameter(block, Type::unsigned(U8_BITS));
        let rhs = dfg.add_block_parameter(block, Type::unsigned(U8_BITS));

        let mut facts = Facts::default();
        facts.set(lhs, ValueRange::Unsigned(lhs_range));
        facts.set(rhs, ValueRange::Unsigned(rhs_range));

        let changed = {
            let back = BinaryBack::new(
                &dfg,
                &mut facts,
                lhs,
                rhs,
                ValueRange::Unsigned(result),
                BinaryRanges::Unsigned(ranges),
            );
            operator.backward(back)
        };

        (
            changed,
            facts.range(lhs).unwrap().into_unsigned().unwrap(),
            facts.range(rhs).unwrap().into_unsigned().unwrap(),
        )
    }

    #[test]
    fn range_not_inverts_bounds_within_type_max() {
        assert_eq!(Range::new(3, 7).not(15), Range::new(8, 12));
    }

    #[test]
    fn range_truncate_wraps_to_full_target_width() {
        assert_eq!(Range::new(0, 10).truncate_to(15), Range::new(0, 10));
        assert_eq!(Range::new(3, 20).truncate_to(15), Range::new(0, 15));
    }

    #[test]
    fn range_intersect_returns_overlap_or_none() {
        assert_eq!(Range::new(2, 8).intersect(Range::new(5, 10)), Some(Range::new(5, 8)));
        assert_eq!(Range::new(2, 8).intersect(Range::new(9, 10)), None);
    }

    #[test]
    fn range_increasing_result_uses_full_range_on_overflow() {
        let range =
            Range::new(200, 254).increasing_result(Range::new(2, 2), 255, u128::checked_add);
        assert_eq!(range, Range::new(0, 255));
    }

    #[test]
    fn signed_range_max_bits_uses_full_width_for_negative_values() {
        assert_eq!(SignedRange::new(0, 7).max_bits(8), 3);
        assert_eq!(SignedRange::new(-1, 7).max_bits(8), 8);
    }

    #[test]
    fn signed_range_add_uses_precise_bounds_or_full_range_on_overflow() {
        let ranges =
            SignedBinaryRanges::new(8, SignedRange::new(-10, 20), SignedRange::new(2, 3)).unwrap();
        assert_eq!(ranges.add(), SignedRange::new(-8, 23));

        let ranges =
            SignedBinaryRanges::new(8, SignedRange::new(120, 127), SignedRange::new(1, 2)).unwrap();
        assert_eq!(ranges.add(), SignedRange::new(-128, 127));
    }

    #[test]
    fn signed_range_mul_checks_all_interval_corners() {
        let ranges =
            SignedBinaryRanges::new(8, SignedRange::new(-4, 6), SignedRange::new(-3, 5)).unwrap();

        assert_eq!(ranges.mul(), SignedRange::new(-20, 30));
    }

    #[test]
    fn signed_range_modulo_tracks_result_sign() {
        let ranges =
            SignedBinaryRanges::new(8, SignedRange::new(-10, 20), SignedRange::new(3, 5)).unwrap();

        assert_eq!(ranges.modulo(), SignedRange::new(-4, 4));
    }

    #[test]
    fn signed_casts_handle_contiguous_twos_complement_ranges() {
        let negative = SignedRange::new(-16, -1);
        assert_eq!(negative.to_unsigned(8, 16), Some(Range::new(240, 255)));
        assert_eq!(
            SignedRange::from_unsigned(Range::new(240, 255), 8),
            Some(SignedRange::new(-16, -1))
        );
    }

    #[test]
    fn recursive_source_uses_cast_fallback_but_facts_source_requires_a_known_range() {
        let mut dfg = DataFlowGraph::default();
        let block = dfg.make_block();
        let original = dfg.add_block_parameter(block, Type::signed(16));
        let result = dfg.add_block_parameter(block, Type::unsigned(8));
        let facts = Facts::default();
        let analysis = Analysis::new(&dfg);

        let recursive_range = analysis.instruction_range(
            &Instruction::Cast(original, NumericType::unsigned(8)),
            result,
            RangeSource::Recursive,
        );
        let facts_range = analysis.instruction_range(
            &Instruction::Cast(original, NumericType::unsigned(8)),
            result,
            RangeSource::Facts(&facts),
        );

        assert_eq!(recursive_range, Some(ValueRange::Unsigned(Range::new(0, 255))));
        assert_eq!(facts_range, None);
    }

    #[test]
    fn field_casts_only_propagate_ranges_from_facts() {
        let mut dfg = DataFlowGraph::default();
        let block = dfg.make_block();
        let original = dfg.add_block_parameter(block, Type::unsigned(8));
        let result = dfg.add_block_parameter(block, Type::field());
        let mut facts = Facts::default();
        facts.set(original, ValueRange::Unsigned(Range::new(3, 7)));
        let analysis = Analysis::new(&dfg);

        let recursive_range = analysis.instruction_range(
            &Instruction::Cast(original, NumericType::NativeField),
            result,
            RangeSource::Recursive,
        );
        let facts_range = analysis.instruction_range(
            &Instruction::Cast(original, NumericType::NativeField),
            result,
            RangeSource::Facts(&facts),
        );

        assert_eq!(recursive_range, None);
        assert_eq!(facts_range, Some(ValueRange::Unsigned(Range::new(3, 7))));
    }

    #[test]
    fn binary_ranges_add_falls_back_when_sum_may_wrap() {
        let ranges = u8_ranges(Range::new(250, 255), Range::new(1, 10));

        assert_eq!(ranges.add(), Range::new(0, 255));
    }

    #[test]
    fn binary_ranges_mul_falls_back_when_product_may_wrap() {
        let ranges = u8_ranges(Range::new(100, 200), Range::new(2, 3));

        assert_eq!(ranges.mul(), Range::new(0, 255));
    }

    #[test]
    fn binary_ranges_sub_distinguishes_checked_and_unchecked_wrap() {
        let ranges = u8_ranges(Range::new(5, 10), Range::new(7, 8));

        assert_eq!(ranges.sub(false), Range::new(0, 3));
        assert_eq!(ranges.sub(true), Range::new(0, 255));
    }

    #[test]
    fn binary_ranges_div_uses_lhs_max_when_rhs_can_be_zero() {
        let ranges = u8_ranges(Range::new(10, 20), Range::new(0, 5));

        assert_eq!(ranges.div(), Range::new(0, 20));
    }

    #[test]
    fn binary_ranges_modulo_uses_lhs_max_when_rhs_can_be_zero() {
        let ranges = u8_ranges(Range::new(10, 20), Range::new(0, 5));

        assert_eq!(ranges.modulo(), Range::new(0, 20));
    }

    #[test]
    fn binary_back_add_refines_both_operands() {
        let (changed, lhs, rhs) = apply_u8_back(
            Range::new(0, 255),
            Range::new(0, 255),
            Range::new(0, 15),
            BinaryOp::Add { unchecked: false },
        );

        assert!(changed);
        assert_eq!(lhs, Range::new(0, 15));
        assert_eq!(rhs, Range::new(0, 15));
    }

    #[test]
    fn binary_back_sub_refines_checked_operands() {
        let (changed, lhs, rhs) = apply_u8_back(
            Range::new(0, 255),
            Range::new(0, 255),
            Range::new(10, 20),
            BinaryOp::Sub { unchecked: false },
        );

        assert!(changed);
        assert_eq!(lhs, Range::new(10, 255));
        assert_eq!(rhs, Range::new(0, 245));
    }

    #[test]
    fn binary_back_mul_refines_positive_operands() {
        let (changed, lhs, rhs) = apply_u8_back(
            Range::new(1, 255),
            Range::new(1, 255),
            Range::new(4, 20),
            BinaryOp::Mul { unchecked: false },
        );

        assert!(changed);
        assert_eq!(lhs, Range::new(1, 20));
        assert_eq!(rhs, Range::new(1, 20));
    }

    #[test]
    fn binary_back_div_refines_lhs_from_nonzero_rhs() {
        let (changed, lhs, rhs) =
            apply_u8_back(Range::new(0, 255), Range::new(2, 10), Range::new(3, 4), BinaryOp::Div);

        assert!(changed);
        assert_eq!(lhs, Range::new(6, 49));
        assert_eq!(rhs, Range::new(2, 10));
    }

    #[test]
    fn binary_back_shl_refines_lhs_for_fixed_shift() {
        let (changed, lhs, rhs) =
            apply_u8_back(Range::new(0, 63), Range::new(2, 2), Range::new(0, 31), BinaryOp::Shl);

        assert!(changed);
        assert_eq!(lhs, Range::new(0, 7));
        assert_eq!(rhs, Range::new(2, 2));
    }
}
