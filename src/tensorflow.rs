use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use ::tensorflow::{DataType, Graph as TensorFlowGraph, Shape, Tensor};
use crate::graph::Graph;
use crate::op::{ConstantValue, Op, OpKind, ScalarType};

pub(crate) fn lower(graph: &Graph) -> Result<TensorFlowGraph, String> {
    let mut tensorflow_graph = TensorFlowGraph::new();
    let mut lowered: HashMap<*const Op, ::tensorflow::Operation> = HashMap::new();
    let output_ptr = Rc::as_ptr(graph.output());
    let filter_inputs = collect_filter_inputs(graph.output());
    for op in graph.operations() {
        let op_ptr = Rc::as_ptr(op);
        let operation = match op.kind() {
            OpKind::Input { name, dtype } => {
                let mut desc = tensorflow_graph.new_operation("Placeholder", name).map_err(|e| e.to_string())?;
                desc.set_attr_type("dtype", data_type(*dtype)).map_err(|e| e.to_string())?;
                let shape = if filter_inputs.contains(&op_ptr) { Shape::new(Some(vec![None])) } else { Shape::new(Some(vec![])) };
                desc.set_attr_shape("shape", &shape).map_err(|e| e.to_string())?;
                desc.finish().map_err(|e| e.to_string())?
            }
            OpKind::Constant { value } => {
                let mut desc = tensorflow_graph.new_operation("Const", &node_name(op_ptr, output_ptr)).map_err(|e| e.to_string())?;
                set_constant(&mut desc, value)?;
                desc.finish().map_err(|e| e.to_string())?
            }
            OpKind::Filter => lower_filter(&mut tensorflow_graph, op, &lowered, &node_name(op_ptr, output_ptr))?,
            _ => {
                let op_type = match op.kind() {
                    OpKind::Add => "Add", OpKind::Sub => "Sub", OpKind::Mul => "Mul", OpKind::Div => "Div", OpKind::Neg => "Neg",
                    OpKind::BitAnd => "BitwiseAnd", OpKind::BitOr => "BitwiseOr", OpKind::BitXor => "BitwiseXor", OpKind::BitwiseNot => "Invert",
                    OpKind::Shl => "LeftShift", OpKind::Shr => "RightShift",
                    OpKind::Equal => "Equal", OpKind::NotEqual => "NotEqual", OpKind::Less => "Less", OpKind::LessEqual => "LessEqual", OpKind::Greater => "Greater", OpKind::GreaterEqual => "GreaterEqual",
                    OpKind::Map => return Err("Map nodes should be eliminated before TensorFlow lowering".into()),
                    OpKind::Input { .. } | OpKind::Constant { .. } | OpKind::Filter => unreachable!(),
                };
                let mut desc = tensorflow_graph.new_operation(op_type, &node_name(op_ptr, output_ptr)).map_err(|e| e.to_string())?;
                for input in op.inputs() {
                    let lowered_input = lowered.get(&Rc::as_ptr(input)).ok_or_else(|| "operation input was not lowered".to_string())?;
                    desc.add_input(lowered_input.clone());
                }
                desc.finish().map_err(|e| e.to_string())?
            }
        };
        lowered.insert(op_ptr, operation);
    }
    Ok(tensorflow_graph)
}

fn lower_filter(graph: &mut TensorFlowGraph, op: &Op, lowered: &HashMap<*const Op, ::tensorflow::Operation>, name: &str) -> Result<::tensorflow::Operation, String> {
    let value = lowered.get(&Rc::as_ptr(&op.inputs()[0])).ok_or_else(|| "filter value was not lowered".to_string())?;
    let predicate = lowered.get(&Rc::as_ptr(&op.inputs()[1])).ok_or_else(|| "filter predicate was not lowered".to_string())?;

    let mut where_desc = graph.new_operation("Where", &format!("{name}_where")).map_err(|e| e.to_string())?;
    where_desc.add_input(predicate.clone());
    let where_op = where_desc.finish().map_err(|e| e.to_string())?;

    let mut squeeze_desc = graph.new_operation("Squeeze", &format!("{name}_squeeze")).map_err(|e| e.to_string())?;
    squeeze_desc.add_input(where_op);
    squeeze_desc.set_attr_int_list("squeeze_dims", &[1]).map_err(|e| e.to_string())?;
    let indices = squeeze_desc.finish().map_err(|e| e.to_string())?;

    let mut axis_desc = graph.new_operation("Const", &format!("{name}_axis")).map_err(|e| e.to_string())?;
    axis_desc.set_attr_type("dtype", DataType::Int32).map_err(|e| e.to_string())?;
    axis_desc.set_attr_tensor("value", Tensor::<i32>::from(0)).map_err(|e| e.to_string())?;
    let axis = axis_desc.finish().map_err(|e| e.to_string())?;

    let mut gather_desc = graph.new_operation("GatherV2", name).map_err(|e| e.to_string())?;
    gather_desc.add_input(value.clone());
    gather_desc.add_input(indices);
    gather_desc.add_input(axis);
    gather_desc.finish().map_err(|e| e.to_string())
}

fn collect_filter_inputs(root: &Rc<Op>) -> HashSet<*const Op> {
    let mut result = HashSet::new();
    collect_filter_inputs_impl(root, false, &mut result);
    result
}
fn collect_filter_inputs_impl(op: &Rc<Op>, under_filter: bool, result: &mut HashSet<*const Op>) {
    let under_filter = under_filter || matches!(op.kind(), OpKind::Filter);
    if under_filter && matches!(op.kind(), OpKind::Input { .. }) { result.insert(Rc::as_ptr(op)); }
    for input in op.inputs() { collect_filter_inputs_impl(input, under_filter, result); }
}
fn data_type(dtype: ScalarType) -> DataType { match dtype { ScalarType::F32 => DataType::Float, ScalarType::F64 => DataType::Double, ScalarType::I32 => DataType::Int32, ScalarType::I64 => DataType::Int64, ScalarType::Bool => DataType::Bool } }
fn set_constant(desc: &mut ::tensorflow::OperationDescription<'_>, value: &ConstantValue) -> Result<(), String> { match value {
    ConstantValue::F32(v) => { desc.set_attr_type("dtype", DataType::Float).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<f32>::from(*v)).map_err(|e| e.to_string())?; }
    ConstantValue::F64(v) => { desc.set_attr_type("dtype", DataType::Double).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<f64>::from(*v)).map_err(|e| e.to_string())?; }
    ConstantValue::I32(v) => { desc.set_attr_type("dtype", DataType::Int32).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<i32>::from(*v)).map_err(|e| e.to_string())?; }
    ConstantValue::I64(v) => { desc.set_attr_type("dtype", DataType::Int64).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<i64>::from(*v)).map_err(|e| e.to_string())?; }
    ConstantValue::Bool(v) => { desc.set_attr_type("dtype", DataType::Bool).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<bool>::from(*v)).map_err(|e| e.to_string())?; }
} Ok(()) }
fn node_name(op_ptr: *const Op, output_ptr: *const Op) -> String { if op_ptr == output_ptr { "output".to_string() } else { format!("graphpack_{:p}", op_ptr) } }
