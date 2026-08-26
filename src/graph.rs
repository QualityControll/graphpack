use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use crate::op::{NodeId, Op};
thread_local! { static ARENA: RefCell<Vec<Rc<Op>>> = const { RefCell::new(Vec::new()) }; }
pub(crate) fn insert(op:Op)->NodeId{ARENA.with(|a|{let mut a=a.borrow_mut();let id=NodeId(a.len());a.push(Rc::new(op));id})}
pub(crate) fn get(id:NodeId)->Rc<Op>{ARENA.with(|a|a.borrow()[id.0].clone())}
#[derive(Clone,Debug)] pub struct Graph{operations:Vec<(NodeId,Rc<Op>)>,output:NodeId}
impl Graph{
 pub(crate) fn from_output(output:NodeId)->Self{let mut operations=Vec::new();let mut visited=HashSet::new();collect(output,&mut visited,&mut operations);Self{operations,output}}
 pub fn operations(&self)->Vec<Rc<Op>>{self.operations.iter().map(|(_,op)|op.clone()).collect()}
 pub fn output(&self)->Rc<Op>{get(self.output)}
 pub(crate) fn output_id(&self)->NodeId{self.output}
 pub(crate) fn operation_nodes(&self)->&[(NodeId,Rc<Op>)]{&self.operations}
 pub fn to_tensorflow(&self)->Result<tensorflow::Graph,String>{crate::tensorflow::lower(self)}
}
fn collect(id:NodeId,visited:&mut HashSet<NodeId>,operations:&mut Vec<(NodeId,Rc<Op>)>){if !visited.insert(id){return}let op=get(id);for &input in op.inputs(){collect(input,visited,operations)}operations.push((id,op));}
