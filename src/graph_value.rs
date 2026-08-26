use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::op::{ConstantValue, Op, OpKind};

/// A typed value flowing through a GraphPack computation graph.
#[derive(Clone)]
pub struct GraphValue<T> {
    pub(crate) op: Rc<Op>,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> GraphValue<T> {
    pub(crate) fn from_op(op: Rc<Op>) -> Self { Self { op, _marker: PhantomData } }
    pub(crate) fn op(&self) -> &Rc<Op> { &self.op }
}

fn constant<T: IntoConstant>(value: T) -> GraphValue<T> {
    GraphValue::from_op(Rc::new(Op::new(
        OpKind::Constant { value: value.into_constant() }, vec![],
    )))
}

trait IntoConstant: Sized { fn into_constant(self) -> ConstantValue; }
impl IntoConstant for f32 { fn into_constant(self) -> ConstantValue { ConstantValue::F32(self) } }
impl IntoConstant for f64 { fn into_constant(self) -> ConstantValue { ConstantValue::F64(self) } }
impl IntoConstant for i32 { fn into_constant(self) -> ConstantValue { ConstantValue::I32(self) } }
impl IntoConstant for i64 { fn into_constant(self) -> ConstantValue { ConstantValue::I64(self) } }
impl IntoConstant for usize { fn into_constant(self) -> ConstantValue { ConstantValue::I64(self as i64) } }
impl IntoConstant for bool { fn into_constant(self) -> ConstantValue { ConstantValue::Bool(self) } }

fn binary_op<T>(lhs: GraphValue<T>, rhs: GraphValue<T>, kind: OpKind) -> GraphValue<T> {
    GraphValue::from_op(Rc::new(Op::new(kind, vec![lhs.op.clone(), rhs.op.clone()])))
}

impl<T> Add for GraphValue<T> { type Output = GraphValue<T>; fn add(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Add) } }
impl<T> Sub for GraphValue<T> { type Output = GraphValue<T>; fn sub(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Sub) } }
impl<T> Mul for GraphValue<T> { type Output = GraphValue<T>; fn mul(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Mul) } }
impl<T> Div for GraphValue<T> { type Output = GraphValue<T>; fn div(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Div) } }
impl<T> Neg for GraphValue<T> { type Output = GraphValue<T>; fn neg(self) -> Self::Output { GraphValue::from_op(Rc::new(Op::new(OpKind::Neg, vec![self.op.clone()]))) } }

macro_rules! impl_scalar_ops {
    ($($t:ty),+) => { $(
        impl Add<$t> for GraphValue<$t> { type Output = GraphValue<$t>; fn add(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Add) } }
        impl Sub<$t> for GraphValue<$t> { type Output = GraphValue<$t>; fn sub(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Sub) } }
        impl Mul<$t> for GraphValue<$t> { type Output = GraphValue<$t>; fn mul(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Mul) } }
        impl Div<$t> for GraphValue<$t> { type Output = GraphValue<$t>; fn div(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Div) } }
    )+ };
}

impl_scalar_ops!(f32, f64, i32, i64, usize, bool);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usize_constant_maps_to_i64() {
        let input = GraphValue::<usize>::from_op(Rc::new(Op::new(OpKind::Input { name: "x".into() }, vec![])));
        let result = input + 42usize;
        assert_eq!(result.op().kind(), &OpKind::Add);
        assert_eq!(result.op().inputs()[1].kind(), &OpKind::Constant { value: ConstantValue::I64(42) });
    }
}
