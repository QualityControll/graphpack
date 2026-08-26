use std::marker::PhantomData;

use crate::graph::Graph;
use crate::graph_value::GraphValue;
use crate::op::{GraphType, Op, OpKind};

#[derive(Clone, Copy)]
pub struct Input<T> {
    node: crate::op::NodeId,
    _marker: PhantomData<fn() -> T>,
}

impl<T: GraphType> Input<T> {
    pub fn new(name: impl Into<String>) -> Self {
        Self { node: crate::graph::insert(Op::new(OpKind::Input { name: name.into(), dtype: T::scalar_type() }, Vec::new())), _marker: PhantomData }
    }
    pub fn map<U, F>(self, f: F) -> GraphValue<U> where F: FnOnce(GraphValue<T>) -> GraphValue<U> { f(GraphValue::from_node(self.node)) }
    pub fn filter<F>(self, predicate: F) -> GraphValue<T> where F: FnOnce(GraphValue<T>) -> GraphValue<bool> {
        let value = GraphValue::from_node(self.node);
        let predicate = predicate(value);
        GraphValue::from_op(Op::new(OpKind::Filter, vec![self.node, predicate.node]))
    }
    pub fn fold<U, F>(self, _init: U, _f: F) -> U where F: FnOnce(U, GraphValue<T>) -> U { todo!("fold graph construction is not implemented yet") }
    pub fn reduce<F>(self, _f: F) -> T where F: FnOnce(GraphValue<T>, GraphValue<T>) -> GraphValue<T> { todo!("reduce graph construction is not implemented yet") }
    pub fn scan<U, F>(self, _init: U, _f: F) -> Input<U> where F: FnOnce(U, GraphValue<T>) -> U { todo!("scan graph construction is not implemented yet") }
    pub fn collect(self) -> tensorflow::Graph { Graph::from_output(self.node).to_tensorflow().expect("failed to lower GraphPack graph to TensorFlow") }
}

pub trait InputTupleMap { type GraphValues; fn map<U, Func>(self, f: Func) -> GraphValue<U> where Func: FnOnce(Self::GraphValues) -> GraphValue<U>; }
macro_rules! impl_input_tuple_map { ($(($($input:ident, $value:ident),+)),+ $(,)?) => { $( impl<$($input: GraphType),+> InputTupleMap for ($(Input<$input>,)+) { type GraphValues = ($(GraphValue<$input>,)+); fn map<U, Func>(self, f: Func) -> GraphValue<U> where Func: FnOnce(Self::GraphValues) -> GraphValue<U> { let ($( $value, )+) = self; f(($(GraphValue::from_node($value.node),)+)) } } )+ }; }
impl_input_tuple_map!((A,a,B,b),(A,a,B,b,C,c),(A,a,B,b,C,c,D,d),(A,a,B,b,C,c,D,d,E,e),(A,a,B,b,C,c,D,d,E,e,F,f),(A,a,B,b,C,c,D,d,E,e,F,f,G,g),(A,a,B,b,C,c,D,d,E,e,F,f,G,g,H,h));
impl<A:GraphType,B:GraphType> InputTupleMap for ((Input<A>,Input<B>),) { type GraphValues=((GraphValue<A>,GraphValue<B>),); fn map<U,F>(self,f:F)->GraphValue<U> where F:FnOnce(Self::GraphValues)->GraphValue<U>{let((a,b),)=self;f(((GraphValue::from_node(a.node),GraphValue::from_node(b.node)),))} }
impl<A:GraphType,B:GraphType,C:GraphType,D:GraphType> InputTupleMap for ((Input<A>,Input<B>),(Input<C>,Input<D>)){type GraphValues=((GraphValue<A>,GraphValue<B>),(GraphValue<C>,GraphValue<D>));fn map<U,F>(self,f:F)->GraphValue<U> where F:FnOnce(Self::GraphValues)->GraphValue<U>{let((a,b),(c,d))=self;f(((GraphValue::from_node(a.node),GraphValue::from_node(b.node)),(GraphValue::from_node(c.node),GraphValue::from_node(d.node))))}}
impl<A:GraphType,B:GraphType,C:GraphType,D:GraphType,E:GraphType,F:GraphType,G:GraphType,H:GraphType> InputTupleMap for ((Input<A>,Input<B>,Input<C>,Input<D>),(Input<E>,Input<F>,Input<G>,Input<H>)){type GraphValues=((GraphValue<A>,GraphValue<B>,GraphValue<C>,GraphValue<D>),(GraphValue<E>,GraphValue<F>,GraphValue<G>,GraphValue<H>));fn map<U,FN>(self,f:FN)->GraphValue<U> where FN:FnOnce(Self::GraphValues)->GraphValue<U>{let((a,b,c,d),(e,f,g,h))=self;f(((GraphValue::from_node(a.node),GraphValue::from_node(b.node),GraphValue::from_node(c.node),GraphValue::from_node(d.node)),(GraphValue::from_node(e.node),GraphValue::from_node(f.node),GraphValue::from_node(g.node),GraphValue::from_node(h.node))))}}
