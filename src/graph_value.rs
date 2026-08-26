use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::op::{Op, OpKind};

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
    use crate::Input;

    #[test]
    fn arithmetic_operators_build_graph_nodes() {
        let x = Input::<f32>::new("x");
        let x_op = x.op().clone();
        let value = GraphValue::from_op(x_op.clone());

        for (kind, result) in [
            (OpKind::Add, value.clone() + value.clone()),
            (OpKind::Sub, value.clone() - value.clone()),
            (OpKind::Mul, value.clone() * value.clone()),
            (OpKind::Div, value.clone() / value.clone()),
        ] {
            assert_eq!(result.op().kind(), &kind);
            assert_eq!(result.op().inputs().len(), 2);
            assert!(Rc::ptr_eq(&result.op().inputs()[0], &x_op));
            assert!(Rc::ptr_eq(&result.op().inputs()[1], &x_op));
        }
    }

    #[test]
    fn neg_builds_graph_node() {
        let x = Input::<f32>::new("x");
        let x_op = x.op().clone();
        let result = -GraphValue::from_op(x_op.clone());

        assert_eq!(result.op().kind(), &OpKind::Neg);
        assert_eq!(result.op().inputs().len(), 1);
        assert!(Rc::ptr_eq(&result.op().inputs()[0], &x_op));
    }
}
