use crate::graph::{self, Graph};
use crate::op::{ConstantValue, NodeId, Op, OpKind, ScalarType};
use num_complex::Complex;
use std::collections::{HashMap, HashSet};
use tensorflow::{DataType, Graph as TensorFlowGraph, Shape, Tensor};

pub(crate) fn lower(graph: &Graph) -> Result<TensorFlowGraph, String> {
    let mut tf = TensorFlowGraph::new();
    let mut lowered: HashMap<NodeId, ::tensorflow::Operation> = HashMap::new();
    let output = graph.output_id();
    let sequence_inputs = collect_sequence_inputs(output);
    for &(id, ref op) in graph.operation_nodes() {
        let operation = match op.kind() {
            OpKind::Input { name, dtype } => {
                let mut d = tf
                    .new_operation("Placeholder", name)
                    .map_err(|e| e.to_string())?;
                d.set_attr_type("dtype", data_type(*dtype))
                    .map_err(|e| e.to_string())?;
                let shape = if sequence_inputs.contains(&id) {
                    Shape::new(Some(vec![None]))
                } else {
                    Shape::new(Some(vec![]))
                };
                d.set_attr_shape("shape", &shape)
                    .map_err(|e| e.to_string())?;
                d.finish().map_err(|e| e.to_string())?
            }
            OpKind::Constant { value } => {
                let mut d = tf
                    .new_operation("Const", &node_name(id, output))
                    .map_err(|e| e.to_string())?;
                set_constant(&mut d, value)?;
                d.finish().map_err(|e| e.to_string())?
            }
            OpKind::Filter => lower_filter(&mut tf, op, &lowered, &node_name(id, output))?,
            OpKind::Take { count } => {
                lower_slice(&mut tf, &lowered, op, &node_name(id, output), 0, *count)?
            }
            OpKind::Skip { count } => lower_slice(
                &mut tf,
                &lowered,
                op,
                &node_name(id, output),
                *count,
                usize::MAX,
            )?,
            OpKind::ReduceSum => {
                lower_reduce(&mut tf, &lowered, op, &node_name(id, output), "Sum")?
            }
            OpKind::ReduceProduct => {
                lower_reduce(&mut tf, &lowered, op, &node_name(id, output), "Prod")?
            }
            OpKind::ReduceMin => {
                lower_reduce(&mut tf, &lowered, op, &node_name(id, output), "Min")?
            }
            OpKind::ReduceMax => {
                lower_reduce(&mut tf, &lowered, op, &node_name(id, output), "Max")?
            }
            OpKind::ReduceAny => {
                lower_reduce(&mut tf, &lowered, op, &node_name(id, output), "Any")?
            }
            OpKind::ReduceAll => {
                lower_reduce(&mut tf, &lowered, op, &node_name(id, output), "All")?
            }
            OpKind::ReduceCount => lower_count(&mut tf, &lowered, op, &node_name(id, output))?,
            OpKind::EnumerateIndex => {
                lower_enumerate_index(&mut tf, &lowered, op, &node_name(id, output))?
            }
            OpKind::TupleGet { index } => {
                lower_tuple_get(&mut tf, &lowered, op, &node_name(id, output), *index)?
            }
            OpKind::ZipLeft | OpKind::ZipRight => {
                lower_tuple_get(&mut tf, &lowered, op, &node_name(id, output), 0)?
            }
            OpKind::Enumerate | OpKind::Zip => {
                return Err("enumerate/zip container nodes should not be lowered directly".into());
            }
            _ => lower_operation(&mut tf, id, op, &lowered, &node_name(id, output))?,
        };
        lowered.insert(id, operation);
    }
    Ok(tf)
}
fn collect_sequence_inputs(root: NodeId) -> HashSet<NodeId> {
    let mut r = HashSet::new();
    fn walk(id: NodeId, sequence: bool, r: &mut HashSet<NodeId>) {
        let op = graph::get(id);
        let sequence = sequence
            || matches!(
                op.kind(),
                OpKind::Filter
                    | OpKind::Take { .. }
                    | OpKind::Skip { .. }
                    | OpKind::Enumerate
                    | OpKind::EnumerateIndex
                    | OpKind::Zip
                    | OpKind::ZipLeft
                    | OpKind::ZipRight
                    | OpKind::TupleGet { .. }
                    | OpKind::ReduceSum
                    | OpKind::ReduceProduct
                    | OpKind::ReduceMin
                    | OpKind::ReduceMax
                    | OpKind::ReduceAny
                    | OpKind::ReduceAll
                    | OpKind::ReduceCount
                    | OpKind::Fold { .. }
                    | OpKind::Reduce { .. }
            );
        if sequence && matches!(op.kind(), OpKind::Input { .. }) {
            r.insert(id);
        }
        for &i in op.inputs() {
            walk(i, sequence, r)
        }
    }
    walk(root, false, &mut r);
    r
}
fn lower_enumerate_index(
    tf: &mut TensorFlowGraph,
    lowered: &HashMap<NodeId, ::tensorflow::Operation>,
    op: &Op,
    name: &str,
) -> Result<::tensorflow::Operation, String> {
    let input = lowered
        .get(&op.inputs()[0])
        .ok_or("enumerate input was not lowered")?
        .clone();
    let mut size = tf
        .new_operation("Size", &format!("{name}_size"))
        .map_err(|e| e.to_string())?;
    size.add_input(input);
    size.set_attr_type("out_type", DataType::Int64)
        .map_err(|e| e.to_string())?;
    let size = size.finish().map_err(|e| e.to_string())?;
    let start = const_i64_scalar(tf, &format!("{name}_start"), 0)?;
    let delta = const_i64_scalar(tf, &format!("{name}_delta"), 1)?;
    let mut range = tf.new_operation("Range", name).map_err(|e| e.to_string())?;
    range.add_input(start);
    range.add_input(size);
    range.add_input(delta);
    range.set_attr_type("Tidx", DataType::Int64)
        .map_err(|e| e.to_string())?;
    range.finish().map_err(|e| e.to_string())
}
fn lower_tuple_get(
    tf: &mut TensorFlowGraph,
    lowered: &HashMap<NodeId, ::tensorflow::Operation>,
    op: &Op,
    name: &str,
    index: usize,
) -> Result<::tensorflow::Operation, String> {
    let input = lowered
        .get(&op.inputs()[0])
        .ok_or("tuple projection input was not lowered")?
        .clone();
    if index == 0 || index == 1 {
        return Ok(input);
    }
    Err(format!("unsupported tuple projection index {index}"))
}
fn lower_operation(
    tf: &mut TensorFlowGraph,
    id: NodeId,
    op: &Op,
    lowered: &HashMap<NodeId, ::tensorflow::Operation>,
    name: &str,
) -> Result<::tensorflow::Operation, String> {
    let typ = match op.kind() {
        OpKind::Add => "Add",
        OpKind::Sub => "Sub",
        OpKind::Mul => "Mul",
        OpKind::Div => "Div",
        OpKind::Neg => "Neg",
        OpKind::BitAnd => "BitwiseAnd",
        OpKind::BitOr => "BitwiseOr",
        OpKind::BitXor => "BitwiseXor",
        OpKind::BitwiseNot => "Invert",
        OpKind::Shl => "LeftShift",
        OpKind::Shr => "RightShift",
        OpKind::Equal => "Equal",
        OpKind::NotEqual => "NotEqual",
        OpKind::Less => "Less",
        OpKind::LessEqual => "LessEqual",
        OpKind::Greater => "Greater",
        OpKind::GreaterEqual => "GreaterEqual",
        OpKind::LogicalAnd => "LogicalAnd",
        OpKind::LogicalOr => "LogicalOr",
        OpKind::LogicalNot => "LogicalNot",
        OpKind::Map => {
            return Err("Map nodes should be eliminated before TensorFlow lowering".into());
        }
        OpKind::Input { .. }
        | OpKind::Constant { .. }
        | OpKind::Filter
        | OpKind::Take { .. }
        | OpKind::Skip { .. }
        | OpKind::Enumerate
        | OpKind::EnumerateIndex
        | OpKind::Zip
        | OpKind::ZipLeft
        | OpKind::ZipRight
        | OpKind::TupleGet { .. }
        | OpKind::ReduceSum
        | OpKind::ReduceProduct
        | OpKind::ReduceMin
        | OpKind::ReduceMax
        | OpKind::ReduceAny
        | OpKind::ReduceAll
        | OpKind::ReduceCount
        | OpKind::Fold { .. }
        | OpKind::Reduce { .. } => unreachable!(),
    };
    let is_u16 = node_scalar_type(id) == Some(ScalarType::U16);
    let is_bool_output = matches!(
        op.kind(),
        OpKind::Equal
            | OpKind::NotEqual
            | OpKind::Less
            | OpKind::LessEqual
            | OpKind::Greater
            | OpKind::GreaterEqual
            | OpKind::LogicalAnd
            | OpKind::LogicalOr
            | OpKind::LogicalNot
    );
    let mut inputs = Vec::with_capacity(op.inputs().len());
    for (index, input) in op.inputs().iter().enumerate() {
        let value = lowered
            .get(input)
            .ok_or("operation input was not lowered")?
            .clone();
        let value =
            if is_u16 && node_scalar_type(*input) == Some(ScalarType::U16) && !is_bool_output {
                cast(
                    tf,
                    value,
                    DataType::Int32,
                    &format!("{name}_cast_in_{index}"),
                )?
            } else {
                value
            };
        inputs.push(value);
    }
    let operation_name = if is_u16 && !is_bool_output {
        format!("{name}_compute")
    } else {
        name.to_string()
    };
    let mut d = tf
        .new_operation(typ, &operation_name)
        .map_err(|e| e.to_string())?;
    for value in inputs {
        d.add_input(value);
    }
    let operation = d.finish().map_err(|e| e.to_string())?;
    if is_u16 && !is_bool_output {
        return cast(tf, operation, DataType::UInt16, name);
    }
    Ok(operation)
}
fn lower_slice(
    tf: &mut TensorFlowGraph,
    lowered: &HashMap<NodeId, ::tensorflow::Operation>,
    op: &Op,
    name: &str,
    start: usize,
    count: usize,
) -> Result<::tensorflow::Operation, String> {
    let input = lowered
        .get(&op.inputs()[0])
        .ok_or("slice input was not lowered")?
        .clone();
    let begin = const_i32_vec(tf, &format!("{name}_begin"), start as i32)?;
    let size = const_i32_vec(
        tf,
        &format!("{name}_size"),
        if count == usize::MAX {
            -1
        } else {
            count as i32
        },
    )?;
    let mut d = tf.new_operation("Slice", name).map_err(|e| e.to_string())?;
    d.add_input(input);
    d.add_input(begin);
    d.add_input(size);
    d.finish().map_err(|e| e.to_string())
}
fn lower_reduce(
    tf: &mut TensorFlowGraph,
    lowered: &HashMap<NodeId, ::tensorflow::Operation>,
    op: &Op,
    name: &str,
    typ: &str,
) -> Result<::tensorflow::Operation, String> {
    let input = lowered
        .get(&op.inputs()[0])
        .ok_or("reduction input was not lowered")?
        .clone();
    let axis = const_i64_scalar(tf, &format!("{name}_axis"), 0)?;
    let mut d = tf.new_operation(typ, name).map_err(|e| e.to_string())?;
    d.add_input(input);
    d.add_input(axis);
    d.set_attr_type("Tidx", DataType::Int64)
        .map_err(|e| e.to_string())?;
    d.set_attr_bool("keep_dims", false)
        .map_err(|e| e.to_string())?;
    d.finish().map_err(|e| e.to_string())
}
fn lower_count(
    tf: &mut TensorFlowGraph,
    lowered: &HashMap<NodeId, ::tensorflow::Operation>,
    op: &Op,
    name: &str,
) -> Result<::tensorflow::Operation, String> {
    let input = lowered
        .get(&op.inputs()[0])
        .ok_or("count input was not lowered")?
        .clone();
    let mut d = tf.new_operation("Size", name).map_err(|e| e.to_string())?;
    d.add_input(input);
    d.set_attr_type("out_type", DataType::Int64)
        .map_err(|e| e.to_string())?;
    d.finish().map_err(|e| e.to_string())
}
fn const_i32_vec(
    tf: &mut TensorFlowGraph,
    name: &str,
    value: i32,
) -> Result<::tensorflow::Operation, String> {
    let mut d = tf.new_operation("Const", name).map_err(|e| e.to_string())?;
    d.set_attr_type("dtype", DataType::Int32)
        .map_err(|e| e.to_string())?;
    d.set_attr_tensor(
        "value",
        Tensor::<i32>::new(&[1])
            .with_values(&[value])
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    d.finish().map_err(|e| e.to_string())
}
fn const_i32_scalar(
    tf: &mut TensorFlowGraph,
    name: &str,
    value: i32,
) -> Result<::tensorflow::Operation, String> {
    let mut d = tf.new_operation("Const", name).map_err(|e| e.to_string())?;
    d.set_attr_type("dtype", DataType::Int32)
        .map_err(|e| e.to_string())?;
    d.set_attr_tensor("value", Tensor::<i32>::from(value))
        .map_err(|e| e.to_string())?;
    d.finish().map_err(|e| e.to_string())
}
fn const_i64_scalar(
    tf: &mut TensorFlowGraph,
    name: &str,
    value: i64,
) -> Result<::tensorflow::Operation, String> {
    let mut d = tf.new_operation("Const", name).map_err(|e| e.to_string())?;
    d.set_attr_type("dtype", DataType::Int64)
        .map_err(|e| e.to_string())?;
    d.set_attr_tensor("value", Tensor::<i64>::from(value))
        .map_err(|e| e.to_string())?;
    d.finish().map_err(|e| e.to_string())
}
fn cast(
    tf: &mut TensorFlowGraph,
    input: ::tensorflow::Operation,
    dtype: DataType,
    name: &str,
) -> Result<::tensorflow::Operation, String> {
    let mut d = tf.new_operation("Cast", name).map_err(|e| e.to_string())?;
    d.add_input(input);
    d.set_attr_type("DstT", dtype).map_err(|e| e.to_string())?;
    d.finish().map_err(|e| e.to_string())
}
fn node_scalar_type(id: NodeId) -> Option<ScalarType> {
    match graph::get(id).kind() {
        OpKind::Input { dtype, .. } => Some(*dtype),
        OpKind::Constant { value } => Some(value.scalar_type()),
        OpKind::Equal
        | OpKind::NotEqual
        | OpKind::Less
        | OpKind::LessEqual
        | OpKind::Greater
        | OpKind::GreaterEqual
        | OpKind::LogicalAnd
        | OpKind::LogicalOr
        | OpKind::LogicalNot
        | OpKind::ReduceAny
        | OpKind::ReduceAll => Some(ScalarType::Bool),
        OpKind::ReduceCount | OpKind::EnumerateIndex => Some(ScalarType::I64),
        OpKind::Add
        | OpKind::Sub
        | OpKind::Mul
        | OpKind::Div
        | OpKind::Neg
        | OpKind::BitAnd
        | OpKind::BitOr
        | OpKind::BitXor
        | OpKind::BitwiseNot
        | OpKind::Shl
        | OpKind::Shr
        | OpKind::Filter
        | OpKind::Map
        | OpKind::Take { .. }
        | OpKind::Skip { .. }
        | OpKind::Enumerate
        | OpKind::Zip
        | OpKind::ZipLeft
        | OpKind::ZipRight
        | OpKind::TupleGet { .. }
        | OpKind::ReduceSum
        | OpKind::ReduceProduct
        | OpKind::ReduceMin
        | OpKind::ReduceMax
        | OpKind::Fold { .. }
        | OpKind::Reduce { .. } => graph::get(id)
            .inputs()
            .first()
            .and_then(|input| node_scalar_type(*input)),
    }
}
fn lower_filter(
    tf: &mut TensorFlowGraph,
    op: &Op,
    lowered: &HashMap<NodeId, ::tensorflow::Operation>,
    name: &str,
) -> Result<::tensorflow::Operation, String> {
    let value = lowered
        .get(&op.inputs()[0])
        .ok_or("filter value was not lowered")?;
    let predicate = lowered
        .get(&op.inputs()[1])
        .ok_or("filter predicate was not lowered")?;
    let mut w = tf
        .new_operation("Where", &format!("{name}_where"))
        .map_err(|e| e.to_string())?;
    w.add_input(predicate.clone());
    let w = w.finish().map_err(|e| e.to_string())?;
    let mut s = tf
        .new_operation("Squeeze", &format!("{name}_squeeze"))
        .map_err(|e| e.to_string())?;
    s.add_input(w);
    s.set_attr_int_list("squeeze_dims", &[1])
        .map_err(|e| e.to_string())?;
    let indices = s.finish().map_err(|e| e.to_string())?;
    let axis = const_i32_scalar(tf, &format!("{name}_axis"), 0)?;
    let mut g = tf
        .new_operation("GatherV2", name)
        .map_err(|e| e.to_string())?;
    g.add_input(value.clone());
    g.add_input(indices);
    g.add_input(axis);
    g.finish().map_err(|e| e.to_string())
}
fn data_type(d: ScalarType) -> DataType {
    match d {
        ScalarType::F32 => DataType::Float,
        ScalarType::F64 => DataType::Double,
        ScalarType::I8 => DataType::Int8,
        ScalarType::U8 => DataType::UInt8,
        ScalarType::I16 => DataType::Int16,
        ScalarType::U16 => DataType::UInt16,
        ScalarType::I32 => DataType::Int32,
        ScalarType::I64 => DataType::Int64,
        ScalarType::Bool => DataType::Bool,
        ScalarType::Complex64 => DataType::Complex64,
        ScalarType::Complex128 => DataType::Complex128,
        ScalarType::String => DataType::String,
    }
}
fn set_constant(
    d: &mut tensorflow::OperationDescription,
    value: &ConstantValue,
) -> Result<(), String> {
    match value {
        ConstantValue::F32(v) => {
            d.set_attr_type("dtype", DataType::Float)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<f32>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::F64(v) => {
            d.set_attr_type("dtype", DataType::Double)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<f64>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::I8(v) => {
            d.set_attr_type("dtype", DataType::Int8)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<i8>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::U8(v) => {
            d.set_attr_type("dtype", DataType::UInt8)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<u8>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::I16(v) => {
            d.set_attr_type("dtype", DataType::Int16)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<i16>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::U16(v) => {
            d.set_attr_type("dtype", DataType::UInt16)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<u16>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::I32(v) => {
            d.set_attr_type("dtype", DataType::Int32)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<i32>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::I64(v) => {
            d.set_attr_type("dtype", DataType::Int64)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<i64>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::Bool(v) => {
            d.set_attr_type("dtype", DataType::Bool)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<bool>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::Complex64(v) => {
            d.set_attr_type("dtype", DataType::Complex64)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<Complex<f32>>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::Complex128(v) => {
            d.set_attr_type("dtype", DataType::Complex128)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<Complex<f64>>::from(*v))
                .map_err(|e| e.to_string())?;
        }
        ConstantValue::String(v) => {
            d.set_attr_type("dtype", DataType::String)
                .map_err(|e| e.to_string())?;
            d.set_attr_tensor("value", Tensor::<String>::from(v.clone()))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
fn node_name(id: NodeId, output: NodeId) -> String {
    if id == output {
        "output".into()
    } else {
        format!("n{}", id.0)
    }
}
