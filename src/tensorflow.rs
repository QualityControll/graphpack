use crate::graph::{self, Graph};
use crate::op::{ConstantValue, Op, OpKind, ScalarType};
use num_complex::Complex;
use std::collections::{HashMap, HashSet};
use tensorflow::{DataType, Graph as TensorFlowGraph, Shape, Tensor};

pub(crate) fn lower(graph: &Graph) -> Result<TensorFlowGraph, String> {
    let mut tensorflow_graph = TensorFlowGraph::new();
    let mut lowered = HashMap::new();
    let output_id = graph.output_id();
    let output_ptr = std::rc::Rc::as_ptr(&graph::get(output_id));
    let filter_inputs = collect_filter_inputs(output_id);
    for op in graph.operations() {
        let op_ptr = std::rc::Rc::as_ptr(op);
        let operation = match op.kind() {
            OpKind::Input { name, dtype } => {
                let mut desc = tensorflow_graph.new_operation("Placeholder", name).map_err(|e| e.to_string())?;
                desc.set_attr_type("dtype", data_type(*dtype)).map_err(|e| e.to_string())?;
                let shape = if filter_inputs.contains(&op_ptr) { Shape::new(Some(vec![None])) } else { Shape::new(Some(vec![])) };
                desc.set_attr_shape("shape", &shape).map_err(|e| e.to_string())?;
                desc.finish().map_err(|e| e.to_string())?
            }
            OpKind::Constant { value } => { let mut desc=tensorflow_graph.new_operation("Const",&node_name(op_ptr,output_ptr)).map_err(|e|e.to_string())?; set_constant(&mut desc,value)?; desc.finish().map_err(|e|e.to_string())? }
            OpKind::Filter => lower_filter(&mut tensorflow_graph,op,&lowered,&node_name(op_ptr,output_ptr))?,
            _ => { let op_type=match op.kind(){OpKind::Add=>"Add",OpKind::Sub=>"Sub",OpKind::Mul=>"Mul",OpKind::Div=>"Div",OpKind::Neg=>"Neg",OpKind::BitAnd=>"BitwiseAnd",OpKind::BitOr=>"BitwiseOr",OpKind::BitXor=>"BitwiseXor",OpKind::BitwiseNot=>"Invert",OpKind::Shl=>"LeftShift",OpKind::Shr=>"RightShift",OpKind::Equal=>"Equal",OpKind::NotEqual=>"NotEqual",OpKind::Less=>"Less",OpKind::LessEqual=>"LessEqual",OpKind::Greater=>"Greater",OpKind::GreaterEqual=>"GreaterEqual",OpKind::LogicalAnd=>"LogicalAnd",OpKind::LogicalOr=>"LogicalOr",OpKind::LogicalNot=>"LogicalNot",OpKind::Map=>return Err("Map nodes should be eliminated before TensorFlow lowering".into()),OpKind::Input{..}|OpKind::Constant{..}|OpKind::Filter=>unreachable!()}; let mut desc=tensorflow_graph.new_operation(op_type,&node_name(op_ptr,output_ptr)).map_err(|e|e.to_string())?; for input in op.inputs(){let lowered_input=lowered.get(&graph::get_id(input)).ok_or_else(||"operation input was not lowered".to_string())?;desc.add_input(lowered_input.clone());} desc.finish().map_err(|e|e.to_string())? }
        };
        lowered.insert(op_ptr,operation);
    }
    Ok(tensorflow_graph)
}
fn lower_filter(graph:&mut TensorFlowGraph,op:&Op,lowered:&HashMap<*const crate::op::Op,::tensorflow::Operation>,name:&str)->Result<::tensorflow::Operation,String>{let value=lowered.get(&op_ptr(&op.inputs()[0])).ok_or("filter value was not lowered")?;let predicate=lowered.get(&op_ptr(&op.inputs()[1])).ok_or("filter predicate was not lowered")?;let mut w=graph.new_operation("Where",&format!("{name}_where")).map_err(|e|e.to_string())?;w.add_input(predicate.clone());let w=w.finish().map_err(|e|e.to_string())?;let mut s=graph.new_operation("Squeeze",&format!("{name}_squeeze")).map_err(|e|e.to_string())?;s.add_input(w);s.set_attr_int_list("squeeze_dims",&[1]).map_err(|e|e.to_string())?;let indices=s.finish().map_err(|e|e.to_string())?;let mut a=graph.new_operation("Const",&format!("{name}_axis")).map_err(|e|e.to_string())?;a.set_attr_type("dtype",DataType::Int32).map_err(|e|e.to_string())?;a.set_attr_tensor("value",Tensor::<i32>::from(0)).map_err(|e|e.to_string())?;let axis=a.finish().map_err(|e|e.to_string())?;let mut g=graph.new_operation("GatherV2",name).map_err(|e|e.to_string())?;g.add_input(value.clone());g.add_input(indices);g.add_input(axis);g.finish().map_err(|e|e.to_string())}
fn op_ptr(id:&crate::op::NodeId)->*const crate::op::Op{std::rc::Rc::as_ptr(&crate::graph::get(*id))}
fn collect_filter_inputs(root:crate::op::NodeId)->HashSet<*const crate::op::Op>{let mut r=HashSet::new();collect_filter_inputs_impl(root,false,&mut r);r}
fn collect_filter_inputs_impl(id:crate::op::NodeId,under:bool,r:&mut HashSet<*const crate::op::Op>){let op=graph::get(id);let under=under||matches!(op.kind(),OpKind::Filter);if under&&matches!(op.kind(),OpKind::Input{..}){r.insert(std::rc::Rc::as_ptr(&op));}for input in op.inputs(){collect_filter_inputs_impl(*input,under,r);}}
fn data_type(dtype:ScalarType)->DataType{match dtype{ScalarType::F32=>DataType::Float,ScalarType::F64=>DataType::Double,ScalarType::I32=>DataType::Int32,ScalarType::I64=>DataType::Int64,ScalarType::Bool=>DataType::Bool,ScalarType::Complex64=>DataType::Complex64,ScalarType::Complex128=>DataType::Complex128,ScalarType::String=>DataType::String}}
fn set_constant(desc:&mut ::tensorflow::OperationDescription<'_>,value:&ConstantValue)->Result<(),String>{match value{ConstantValue::F32(v)=>{desc.set_attr_type("dtype",DataType::Float).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<f32>::from(*v)).map_err(|e|e.to_string())?},ConstantValue::F64(v)=>{desc.set_attr_type("dtype",DataType::Double).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<f64>::from(*v)).map_err(|e|e.to_string())?},ConstantValue::I32(v)=>{desc.set_attr_type("dtype",DataType::Int32).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<i32>::from(*v)).map_err(|e|e.to_string())?},ConstantValue::I64(v)=>{desc.set_attr_type("dtype",DataType::Int64).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<i64>::from(*v)).map_err(|e|e.to_string())?},ConstantValue::Bool(v)=>{desc.set_attr_type("dtype",DataType::Bool).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<bool>::from(*v)).map_err(|e|e.to_string())?},ConstantValue::Complex64(v)=>{desc.set_attr_type("dtype",DataType::Complex64).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<Complex<f32>>::from(*v)).map_err(|e|e.to_string())?},ConstantValue::Complex128(v)=>{desc.set_attr_type("dtype",DataType::Complex128).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<Complex<f64>>::from(*v)).map_err(|e|e.to_string())?},ConstantValue::String(v)=>{desc.set_attr_type("dtype",DataType::String).map_err(|e|e.to_string())?;desc.set_attr_tensor("value",Tensor::<String>::from(v.clone())).map_err(|e|e.to_string())?}}Ok(())}
fn node_name(op_ptr:*const crate::op::Op,output_ptr:*const crate::op::Op)->String{if op_ptr==output_ptr{"output".into()}else{format!("graphpack_{:p}",op_ptr)}}
