use std::marker::PhantomData;
use crate::graph::Graph;
use crate::graph_value::GraphValue;
use crate::op::{GraphType, Op, OpKind, ReduceKind};

#[derive(Clone, Copy)] pub struct Input<T> { node: crate::op::NodeId, _marker: PhantomData<fn() -> T> }
#[derive(Clone, Copy)] pub struct GraphSeq<T> { pub(crate) node: crate::op::NodeId, _marker: PhantomData<fn() -> T> }
impl<T> GraphSeq<T> { pub(crate) fn from_node(node:crate::op::NodeId)->Self{Self{node,_marker:PhantomData}} pub fn collect(&self)->tensorflow::Graph{Graph::from_output(self.node).to_tensorflow().expect("failed to lower GraphPack graph to TensorFlow")} }
impl<T:GraphType> Input<T> {
 pub fn new(name: impl Into<String>) -> Self { Self { node: crate::graph::insert(Op::new(OpKind::Input{name:name.into(),dtype:T::scalar_type()},Vec::new())), _marker:PhantomData } }
 pub fn map<U,F>(self,f:F)->GraphValue<U> where F:FnOnce(GraphValue<T>)->GraphValue<U>{f(GraphValue::from_node(self.node))}
 pub fn sequence(self)->GraphSeq<T>{GraphSeq::from_node(self.node)}
 pub fn filter<F>(self,predicate:F)->GraphSeq<T> where F:FnOnce(GraphValue<T>)->GraphValue<bool>{let p=predicate(GraphValue::from_node(self.node));GraphSeq::from_node(crate::graph::insert(Op::new(OpKind::Filter,vec![self.node,p.node])))}
 pub fn take(self,count:usize)->GraphSeq<T>{GraphSeq::from_node(crate::graph::insert(Op::new(OpKind::Take{count},vec![self.node])))}
 pub fn skip(self,count:usize)->GraphSeq<T>{GraphSeq::from_node(crate::graph::insert(Op::new(OpKind::Skip{count},vec![self.node])))}
 pub fn sum(self)->GraphValue<T>{GraphValue::from_node(crate::graph::insert(Op::new(OpKind::ReduceSum,vec![self.node])))}
 pub fn product(self)->GraphValue<T>{GraphValue::from_node(crate::graph::insert(Op::new(OpKind::ReduceProduct,vec![self.node])))}
 pub fn min(self)->GraphValue<T>{GraphValue::from_node(crate::graph::insert(Op::new(OpKind::ReduceMin,vec![self.node])))}
 pub fn max(self)->GraphValue<T>{GraphValue::from_node(crate::graph::insert(Op::new(OpKind::ReduceMax,vec![self.node])))}
 pub fn count(self)->GraphValue<i64>{GraphValue::from_node(crate::graph::insert(Op::new(OpKind::ReduceCount,vec![self.node])))}
 pub fn fold<U,F>(self,_init:U,_f:F)->GraphValue<U> where F:FnOnce(GraphValue<U>,GraphValue<T>)->GraphValue<U>{todo!("fold lowering is not implemented yet")}
 pub fn reduce<F>(self,_f:F)->GraphValue<T> where F:FnOnce(GraphValue<T>,GraphValue<T>)->GraphValue<T>{todo!("reduce lowering is not implemented yet")}
}
impl<T> GraphSeq<T>{
 pub fn take(self,count:usize)->Self{Self::from_node(crate::graph::insert(Op::new(OpKind::Take{count},vec![self.node])))}
 pub fn skip(self,count:usize)->Self{Self::from_node(crate::graph::insert(Op::new(OpKind::Skip{count},vec![self.node])))}
 pub fn filter<F>(self,_predicate:F)->Self{todo!("filter on a GraphSeq is not implemented yet")}
 pub fn enumerate(self)->GraphSeq<(i64,T)>{Self::from_node(crate::graph::insert(Op::new(OpKind::Enumerate,vec![self.node])))}
 pub fn zip<U>(self,other:GraphSeq<U>)->GraphSeq<(T,U)>{Self::from_node(crate::graph::insert(Op::new(OpKind::Zip,vec![self.node,other.node])))}
}
impl<T:GraphType> Input<T>{pub fn collect(self)->tensorflow::Graph{Graph::from_output(self.node).to_tensorflow().expect("failed to lower GraphPack graph to TensorFlow")}}

pub trait InputTupleMap { type GraphValues; fn map<U,Func>(self,f:Func)->GraphValue<U> where Func:FnOnce(Self::GraphValues)->GraphValue<U>; }
macro_rules! impl_input_tuple_map { ($(($($input:ident, $value:ident),+)),+ $(,)?) => { $( impl<$($input: GraphType),+> InputTupleMap for ($(Input<$input>,)+) { type GraphValues = ($(GraphValue<$input>,)+); fn map<U, Func>(self, func: Func) -> GraphValue<U> where Func: FnOnce(Self::GraphValues) -> GraphValue<U> { let ($( $value, )+) = self; func(($(GraphValue::from_node($value.node),)+)) } } )+ }; }
impl_input_tuple_map!((A,a,B,b),(A,a,B,b,C,c),(A,a,B,b,C,c,D,d),(A,a,B,b,C,c,D,d,E,e),(A,a,B,b,C,c,D,d,E,e,F,f),(A,a,B,b,C,c,D,d,E,e,F,f,G,g),(A,a,B,b,C,c,D,d,E,e,F,f,G,g,H,h));
impl<A:GraphType,B:GraphType> InputTupleMap for ((Input<A>,Input<B>),) { type GraphValues=((GraphValue<A>,GraphValue<B>),); fn map<U,F>(self,func:F)->GraphValue<U> where F:FnOnce(Self::GraphValues)->GraphValue<U>{let((a,b),)=self;func(((GraphValue::from_node(a.node),GraphValue::from_node(b.node)),))} }
impl<A:GraphType,B:GraphType,C:GraphType,D:GraphType> InputTupleMap for ((Input<A>,Input<B>),(Input<C>,Input<D>)){type GraphValues=((GraphValue<A>,GraphValue<B>),(GraphValue<C>,GraphValue<D>));fn map<U,F>(self,func:F)->GraphValue<U> where F:FnOnce(Self::GraphValues)->GraphValue<U>{let((a,b),(c,d))=self;func(((GraphValue::from_node(a.node),GraphValue::from_node(b.node)),(GraphValue::from_node(c.node),GraphValue::from_node(d.node))))}}
impl<A:GraphType,B:GraphType,C:GraphType,D:GraphType,E:GraphType,F:GraphType,G:GraphType,H:GraphType> InputTupleMap for ((Input<A>,Input<B>,Input<C>,Input<D>),(Input<E>,Input<F>,Input<G>,Input<H>)){type GraphValues=((GraphValue<A>,GraphValue<B>,GraphValue<C>,GraphValue<D>),(GraphValue<E>,GraphValue<F>,GraphValue<G>,GraphValue<H>));fn map<U,FN>(self,func:FN)->GraphValue<U> where FN:FnOnce(Self::GraphValues)->GraphValue<U>{let((a,b,c,d),(e,f,g,h))=self;func(((GraphValue::from_node(a.node),GraphValue::from_node(b.node),GraphValue::from_node(c.node),GraphValue::from_node(d.node)),(GraphValue::from_node(e.node),GraphValue::from_node(f.node),GraphValue::from_node(g.node),GraphValue::from_node(h.node))))}}
