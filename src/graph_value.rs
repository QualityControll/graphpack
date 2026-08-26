use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Shl, Shr, Sub};
use std::rc::Rc;

use crate::graph::Graph;
use crate::op::{ConstantValue, Op, OpKind};

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

    pub fn map<U, F>(self, f: F) -> GraphValue<U>
    where
        F: FnOnce(GraphValue<T>) -> GraphValue<U>,
    {
        f(self)
    }

    pub fn collect(&self) -> tensorflow::Graph {
        Graph::from_output(self.op.clone())
            .to_tensorflow()
            .expect("failed to lower GraphPack graph to TensorFlow")
    }

    pub fn eq(self, rhs: GraphValue<T>) -> GraphValue<bool> {
        comparison(self, rhs, OpKind::Equal)
    }

    pub fn ne(self, rhs: GraphValue<T>) -> GraphValue<bool> {
        comparison(self, rhs, OpKind::NotEqual)
    }

    pub fn lt(self, rhs: GraphValue<T>) -> GraphValue<bool> {
        comparison(self, rhs, OpKind::Less)
    }

    pub fn le(self, rhs: GraphValue<T>) -> GraphValue<bool> {
        comparison(self, rhs, OpKind::LessEqual)
    }

    pub fn gt(self, rhs: GraphValue<T>) -> GraphValue<bool> {
        comparison(self, rhs, OpKind::Greater)
    }

    pub fn ge(self, rhs: GraphValue<T>) -> GraphValue<bool> {
        comparison(self, rhs, OpKind::GreaterEqual)
    }
}

fn constant<T: IntoConstant>(value: T) -> GraphValue<T> {
    GraphValue::from_op(Rc::new(Op::new(
        OpKind::Constant { value: value.into_constant() },
        vec![],
    )))
}

trait IntoConstant: Sized {
    fn into_constant(self) -> ConstantValue;
}

impl IntoConstant for f32 {
    fn into_constant(self) -> ConstantValue { ConstantValue::F32(self) }
}
impl IntoConstant for f64 {
    fn into_constant(self) -> ConstantValue { ConstantValue::F64(self) }
}
impl IntoConstant for i32 {
    fn into_constant(self) -> ConstantValue { ConstantValue::I32(self) }
}
impl IntoConstant for i64 {
    fn into_constant(self) -> ConstantValue { ConstantValue::I64(self) }
}
impl IntoConstant for usize {
    fn into_constant(self) -> ConstantValue { ConstantValue::I64(self as i64) }
}
impl IntoConstant for bool {
    fn into_constant(self) -> ConstantValue { ConstantValue::Bool(self) }
}

fn binary_op<T>(lhs: GraphValue<T>, rhs: GraphValue<T>, kind: OpKind) -> GraphValue<T> {
    GraphValue::from_op(Rc::new(Op::new(
        kind,
        vec![lhs.op.clone(), rhs.op.clone()],
    )))
}

fn unary_op<T>(value: GraphValue<T>, kind: OpKind) -> GraphValue<T> {
    GraphValue::from_op(Rc::new(Op::new(kind, vec![value.op.clone()])))
}

fn comparison<T>(lhs: GraphValue<T>, rhs: GraphValue<T>, kind: OpKind) -> GraphValue<bool> {
    GraphValue::from_op(Rc::new(Op::new(
        kind,
        vec![lhs.op.clone(), rhs.op.clone()],
    )))
}

macro_rules! impl_comparison_ops {
    ($($t:ty),+) => {
        $(
            impl GraphValue<$t> {
                pub fn eq_scalar(self, rhs: $t) -> GraphValue<bool> {
                    comparison(self, constant(rhs), OpKind::Equal)
                }
                pub fn ne_scalar(self, rhs: $t) -> GraphValue<bool> {
                    comparison(self, constant(rhs), OpKind::NotEqual)
                }
                pub fn lt_scalar(self, rhs: $t) -> GraphValue<bool> {
                    comparison(self, constant(rhs), OpKind::Less)
                }
                pub fn le_scalar(self, rhs: $t) -> GraphValue<bool> {
                    comparison(self, constant(rhs), OpKind::LessEqual)
                }
                pub fn gt_scalar(self, rhs: $t) -> GraphValue<bool> {
                    comparison(self, constant(rhs), OpKind::Greater)
                }
                pub fn ge_scalar(self, rhs: $t) -> GraphValue<bool> {
                    comparison(self, constant(rhs), OpKind::GreaterEqual)
                }
            }
        )+
    };
}

impl_comparison_ops!(f32, f64, i32, i64, usize);

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
    fn neg(self) -> Self::Output { unary_op(self, OpKind::Neg) }
}

macro_rules! impl_scalar_ops {
    ($($t:ty),+) => {
        $(
            impl Add<$t> for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn add(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Add) }
            }
            impl Sub<$t> for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn sub(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Sub) }
            }
            impl Mul<$t> for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn mul(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Mul) }
            }
            impl Div<$t> for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn div(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::Div) }
            }
        )+
    };
}
impl_scalar_ops!(f32, f64, i32, i64, usize, bool);

macro_rules! impl_bitwise_ops {
    ($($t:ty),+) => {
        $(
            impl BitAnd for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn bitand(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::BitAnd) }
            }
            impl BitOr for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn bitor(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::BitOr) }
            }
            impl BitXor for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn bitxor(self, rhs: Self) -> Self::Output { binary_op(self, rhs, OpKind::BitXor) }
            }
            impl Not for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn not(self) -> Self::Output { unary_op(self, OpKind::BitwiseNot) }
            }
            impl BitAnd<$t> for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn bitand(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::BitAnd) }
            }
            impl BitOr<$t> for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn bitor(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::BitOr) }
            }
            impl BitXor<$t> for GraphValue<$t> {
                type Output = GraphValue<$t>;
                fn bitxor(self, rhs: $t) -> Self::Output { binary_op(self, constant(rhs), OpKind::BitXor) }
            }
        )+
    };
}
impl_bitwise_ops!(i32, i64);

impl Shl<u32> for GraphValue<i32> {
    type Output = GraphValue<i32>;
    fn shl(self, rhs: u32) -> Self::Output { binary_op(self, constant(rhs as i32), OpKind::Shl) }
}
impl Shr<u32> for GraphValue<i32> {
    type Output = GraphValue<i32>;
    fn shr(self, rhs: u32) -> Self::Output { binary_op(self, constant(rhs as i32), OpKind::Shr) }
}
impl Shl<u32> for GraphValue<i64> {
    type Output = GraphValue<i64>;
    fn shl(self, rhs: u32) -> Self::Output { binary_op(self, constant(rhs as i64), OpKind::Shl) }
}
impl Shr<u32> for GraphValue<i64> {
    type Output = GraphValue<i64>;
    fn shr(self, rhs: u32) -> Self::Output { binary_op(self, constant(rhs as i64), OpKind::Shr) }
}
