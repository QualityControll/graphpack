use crate::graph::{self, Graph};
use crate::op::{ConstantValue, Op, OpKind, NodeId, ScalarType};
use num_complex::Complex;
use std::collections::{HashMap,HashSet};
use tensorflow::{DataType,Graph as TensorFlowGraph,Shape,Tensor};

pub(crate) fn lower(graph:&Graph)->Result<TensorFlowGraph,String>{
 let mut tf=TensorFlowGraph::new();
 let mut lowered:HashMap<NodeId,::tensorflow::Operation>=HashMap::new();
 let output=graph.output_id();
 let filters=collect_filter_inputs(output);
 for &(id,ref op) in graph.operation_nodes(){
  let operation=match op.kind(){
   OpKind::Input{name,dtype}=>{let mut d=tf.new_operation("Placeholder",name).map_err(|e|e.to_string())?;d.set_attr_type("dtype",data_type(*dtype)).map_err(|e|e.to_string())?;d.set_attr_shape("shape",&if filters.contains(&id){Shape::new(Some(vec![None]))}else{Shape::new(Some(vec![]))}).map_err(|e|e.to_string())?;d.finish().map_err(|e|e.to_string())?}
   OpKind::Constant{value}=>{let mut d=tf.new_operation("Const",&node_name(id,output)).map_err(|e|e.to_string())?;set_constant(&mut d,value)?;d.finish().map_err(|e|e.to_string())?}
   OpKind::Filter=>lower_filter(&mut tf,op,&lowered,&node_name(id,output))?,
   _=>lower_operation(&mut tf,id,op,&lowered,&node_name(id,output))?
  };
  lowered.insert(id,operation);
 }
 Ok(tf)
}

fn lower_operation(tf:&mut TensorFlowGraph,id:NodeId,op:&Op,lowered:&HashMap<NodeId,::tensorflow::Operation>,name:&str)->Result<::tensorflow::Operation,String>{
 let typ=match op.kind(){OpKind::Add=>"Add",OpKind::Sub=>"Sub",OpKind::Mul=>"Mul",OpKind::Div=>"Div",OpKind::Neg=>"Neg",OpKind::BitAnd=>"BitwiseAnd",OpKind::BitOr=>"BitwiseOr",OpKind::BitXor=>"BitwiseXor",OpKind::BitwiseNot=>"Invert",OpKind::Shl=>"LeftShift",OpKind::Shr=>"RightShift",OpKind::Equal=>"Equal",OpKind::NotEqual=>"NotEqual",OpKind::Less=>"Less",OpKind::LessEqual=>"LessEqual",OpKind::Greater=>"Greater",OpKind::GreaterEqual=>"GreaterEqual",OpKind::LogicalAnd=>"LogicalAnd",OpKind::LogicalOr=>"LogicalOr",OpKind::LogicalNot=>"LogicalNot",OpKind::Map=>return Err("Map nodes should be eliminated before TensorFlow lowering".into()),OpKind::Input{..}|OpKind::Constant{..}|OpKind::Filter=>unreachable!()};
 let mut d=tf.new_operation(typ,name).map_err(|e|e.to_string())?;
 for input in op.inputs(){d.add_input(lowered.get(input).ok_or("operation input was not lowered")?.clone());}
 let operation=d.finish().map_err(|e|e.to_string())?;
 if matches!(op.kind(),OpKind::Add|OpKind::Sub|OpKind::Mul) && node_scalar_type(op.inputs()[0])==Some(ScalarType::U16){
  let mut mask=tf.new_operation("Const",&format!("{name}_u16_mask")).map_err(|e|e.to_string())?;
  mask.set_attr_type("dtype",DataType::Int32).map_err(|e|e.to_string())?;
  mask.set_attr_tensor("value",Tensor::<i32>::from(0xffff)).map_err(|e|e.to_string())?;
  let mask=mask.finish().map_err(|e|e.to_string())?;
  let mut and=tf.new_operation("BitwiseAnd",&format!("{name}_u16_wrap")).map_err(|e|e.to_string())?;
  and.add_input(operation);
  and.add_input(mask);
  return and.finish().map_err(|e|e.to_string());
 }
 Ok(operation)
}

fn node_scalar_type(id:NodeId)->Option<ScalarType>{match graph::get(id).kind(){OpKind::Input{dtype,..}=>Some(*dtype),OpKind::Constant{value}=>Some(value.scalar_type()),OpKind::Equal|OpKind::NotEqual|OpKind::Less|OpKind::LessEqual|OpKind::Greater|OpKind::GreaterEqual|OpKind::LogicalAnd|OpKind::LogicalOr|OpKind::LogicalNot=>Some(ScalarType::Bool),OpKind::Add|OpKind::Sub|OpKind::Mul|OpKind::Div|OpKind::Neg|OpKind::BitAnd|OpKind::BitOr|OpKind::BitXor|OpKind::BitwiseNot|OpKind::Shl|OpKind::Shr|OpKind::Filter|OpKind::Map=>graph::get(id).inputs().first().and_then(|input|node_scalar_type(*input))}}

fn lower_filter(tf:&mut TensorFlowGraph,op:&Op,lowered:&HashMap<NodeId,::tensorflow::Operation>,name:&str)->Result<::tensorflow::Operation,String>{let value=lowered.get(&op.inputs()[0]).ok_or("filter value was not lowered")?;let predicate=lowered.get(&op.inputs()[1]).ok_or("filter predicate was not lowered")?;let mut w=tf.new_operation("Where",&format!("{name}_where")).map_err(|e|e.to_string())?;w.add_input(predicate.clone());let w=w.finish().map_err(|e|e.to_string())?;let mut s=tf.new_operation("Squeeze",&format!("{name}_squeeze")).map_err(|e|e.to_string())?;s.add_input(w);s.set_attr_int_list("squeeze_dims",&[1]).map_err(|e|e.to_string())?;let indices=s.finish().map_err(|e|e.to_string())?;let mut a=tf.new_operation("Const",&format!("{name}_axis")).map_err(|e|e.to_string())?;a.set_attr_type("dtype",DataType::Int32).map_err(|e|e.to_string())?;a.set_attr_tensor("value",Tensor::<i32>::from(0)).map_err(|e|e.to_string())?;let axis=a.finish().map_err(|e|e.to_string())?;let mut g=tf.new_operation("GatherV2",name).map_err(|e|e.to_string())?;g.add_input(value.clone());g.add_input(indices);g.add_input(axis);g.finish().map_err(|e|e.to_string())}
fn collect_filter_inputs(root:NodeId)->HashSet<NodeId>{let mut r=HashSet::new();fn walk(id:NodeId,under:bool,r:&mut HashSet<NodeId>){let op=graph::get(id);let under=under||matches!(op.kind(),OpKind::Filter);if under&&matches!(op.kind(),OpKind::Input{..}){r.insert(id);}for &i in op.inputs(){walk(i,under,r)}}walk(root,false,&mut r);r}
fn data_type(d:ScalarType)->DataType{match d{ScalarType::F32=>DataType::Float,ScalarType::F64=>DataType::Double,ScalarType::I8=>DataType::Int8,ScalarType::U8=>DataType::UInt8,ScalarType::I16=>DataType::Int16,ScalarType::U16=>DataType::Int32,ScalarType::I32=>DataType::Int32,ScalarType::I64=>DataType::Int64,ScalarType::Bool=>DataType::Bool,ScalarType::Complex64=>DataType::Complex64,ScalarType::Complex128=>DataType::Complex128,ScalarType::String=>DataType::String}}
fn set_constant(d:&mut ::tensorflow::OperationDescription<'_>,v:&ConstantValue)->Result<(),String>{match v{ConstantValue::F32(x)=>{d.set_attr_type("dtype",DataType::Float).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<f32>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::F64(x)=>{d.set_attr_type("dtype",DataType::Double).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<f64>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::I8(x)=>{d.set_attr_type("dtype",DataType::Int8).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<i8>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::U8(x)=>{d.set_attr_type("dtype",DataType::UInt8).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<u8>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::I16(x)=>{d.set_attr_type("dtype",DataType::Int16).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<i16>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::U16(x)=>{d.set_attr_type("dtype",DataType::Int32).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<i32>::from(*x as i32)).map_err(|e|e.to_string())?},ConstantValue::I32(x)=>{d.set_attr_type("dtype",DataType::Int32).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<i32>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::I64(x)=>{d.set_attr_type("dtype",DataType::Int64).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<i64>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::Bool(x)=>{d.set_attr_type("dtype",DataType::Bool).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<bool>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::Complex64(x)=>{d.set_attr_type("dtype",DataType::Complex64).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<Complex<f32>>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::Complex128(x)=>{d.set_attr_type("dtype",DataType::Complex128).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<Complex<f64>>::from(*x)).map_err(|e|e.to_string())?},ConstantValue::String(x)=>{d.set_attr_type("dtype",DataType::String).map_err(|e|e.to_string())?;d.set_attr_tensor("value",Tensor::<String>::from(x.clone())).map_err(|e|e.to_string())?}}Ok(())}
fn node_name(id:NodeId,output:NodeId)->String{if id==output{"output".into()}else{format!("graphpack_{}",id.0)}}
