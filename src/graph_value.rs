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
    pub(crate) fn from_op(op: Rc<Op>) -> Self {
        Self { op, _marker: PhantomData }
    }

    pub(crate) fn op(&self) -> &Rc<Op> {
        &self.op
    }
}

impl GraphValue<f32> {
    pub fn constant(value: f32) -> Self {
        Self::from_op(Rc::new(Op::new(OpKind::Constant { value: ConstantValue::F32(value) }, vec![])))
    }
}

impl GraphValue<f64> {
    pub fn constant(value: f64) -> Self {
        Self::from_op(Rc::new(Op::new(OpKind::Constant { value: ConstantValue::F64(value) }, vec![])))
    }
}

impl GraphValue<i32> {
    pub fn constant(value: i32) -> Self {
        Self::from_op(Rc::new(Op::new(OpKind::Constant { value: ConstantValue::I32(value) }, vec![])))
    }
}

impl GraphValue<i64> {
    pub fn constant(value: i64) -> Self {
        Self::from_op(Rc::new(Op::new(OpKind::Constant { value: ConstantValue::I64(value) }, vec![])))
    }
}

impl GraphValue<bool> {
    pub fn constant(value: bool) -> Self {
        Self::from_op(Rc::new(Op::new(OpKind::Constant { value: ConstantValue::Bool(value) }, vec![])))
    }
}

fn binary_op<T>(lhs: GraphValue<T>, rhs: GraphValue<T>, kind: OpKind) -> GraphValue<T> {
    GraphValue::from_op(Rc::new(Op::new(kind, vec![lhs.op.clone(), rhs.op.clone()])))
}

impl<T> Add for GraphValue<T> {
    type Output = GraphValue<T>;
    fn add(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Add) }
}

impl<T> Sub for GraphValue<T> {
    type Output = GraphValue<T>;
    fn sub(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Sub) }
}

impl<T> Mul for GraphValue<T> {
    type Output = GraphValue<T>;
    fn mul(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Mul) }
}

impl<T> Div for GraphValue<T> {
    type Output = GraphValue<T>;
    fn div(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::Div) }
}

impl<T> Neg for GraphValue<T> {
    type Output = GraphValue<T>;
    fn neg(self) -> Self::Output {
        GraphValue::from_op(Rc::new(Op::new(OpKind::Neg, vec![self.op.clone()])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_build_graph_nodes() {
        let f32_value = GraphValue::<f32>::constant(2.5);
        assert_eq!(f32_value.op().kind(), &OpKind::Constant { value: ConstantValue::F32(2.5) });
        assert!(f32_value.op().inputs().is_empty());

        let i32_value = GraphValue::<i32>::constant(7);
        assert_eq!(i32_value.op().kind(), &OpKind::Constant { value: ConstantValue::I32(7) });
    }
}
