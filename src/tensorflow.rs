use std::collections::HashMap;
use std::rc::Rc;

use tensorflow::{DataType, Graph as TensorFlowGraph, Shape, Tensor};

use crate::graph::Graph;
use crate::op::{ConstantValue, Op, OpKind, ScalarType};

pub(crate) fn lower(graph: &Graph) -> Result<TensorFlowGraph, String> {
    let mut tensorflow_graph = TensorFlowGraph::new();
    let mut lowered: HashMap<*const Op, tensorflow::Operation> = HashMap::new();
    let output_ptr = Rc::as_ptr(graph.output());

    for op in graph.operations() {
        let op_ptr = Rc::as_ptr(op);
        let operation = match op.kind() {
            OpKind::Input { name, dtype } => {
                let mut desc = tensorflow_graph.new_operation("Placeholder", name).map_err(|e| e.to_string())?;
                desc.set_attr_type("dtype", data_type(*dtype)).map_err(|e| e.to_string())?;
                desc.set_attr_shape("shape", &Shape(Some(vec![]))).map_err(|e| e.to_string())?;
                desc.finish().map_err(|e| e.to_string())?
            }
            OpKind::Constant { value } => {
                let mut desc = tensorflow_graph.new_operation("Const", &node_name(op_ptr, output_ptr)).map_err(|e| e.to_string())?;
                set_constant(&mut desc, value)?;
                desc.finish().map_err(|e| e.to_string())?
            }
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Neg => {
                let op_type = match op.kind() {
                    OpKind::Add => "Add", OpKind::Sub => "Sub", OpKind::Mul => "Mul", OpKind::Div => "Div", OpKind::Neg => "Neg", _ => unreachable!(),
                };
                let mut desc = tensorflow_graph.new_operation(op_type, &node_name(op_ptr, output_ptr)).map_err(|e| e.to_string())?;
                for input in op.inputs() {
                    let lowered_input = lowered.get(&Rc::as_ptr(input)).ok_or_else(|| "operation input was not lowered".to_string())?;
                    desc.add_input(lowered_input.clone());
                }
                desc.finish().map_err(|e| e.to_string())?
            }
            OpKind::Map => return Err("Map nodes should be eliminated before TensorFlow lowering".into()),
        };
        lowered.insert(op_ptr, operation);
    }

    Ok(tensorflow_graph)
}

fn data_type(dtype: ScalarType) -> DataType {
    match dtype { ScalarType::F32 => DataType::Float, ScalarType::F64 => DataType::Double, ScalarType::I32 => DataType::Int32, ScalarType::I64 => DataType::Int64, ScalarType::Bool => DataType::Bool }
}

fn set_constant(desc: &mut tensorflow::OperationDescription<'_>, value: &ConstantValue) -> Result<(), String> {
    match value {
        ConstantValue::F32(value) => { desc.set_attr_type("dtype", DataType::Float).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<f32>::from(*value)).map_err(|e| e.to_string())?; }
        ConstantValue::F64(value) => { desc.set_attr_type("dtype", DataType::Double).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<f64>::from(*value)).map_err(|e| e.to_string())?; }
        ConstantValue::I32(value) => { desc.set_attr_type("dtype", DataType::Int32).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<i32>::from(*value)).map_err(|e| e.to_string())?; }
        ConstantValue::I64(value) => { desc.set_attr_type("dtype", DataType::Int64).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<i64>::from(*value)).map_err(|e| e.to_string())?; }
        ConstantValue::Bool(value) => { desc.set_attr_type("dtype", DataType::Bool).map_err(|e| e.to_string())?; desc.set_attr_tensor("value", Tensor::<bool>::from(*value)).map_err(|e| e.to_string())?; }
    }
    Ok(())
}

fn node_name(op_ptr: *const Op, output_ptr: *const Op) -> String {
    if op_ptr == output_ptr { "output".to_string() } else { format!("graphpack_{:p}", op_ptr) }
}
