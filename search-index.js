var searchIndex = {};
searchIndex["menoh"] = {"doc":"Menoh-rs A simple wrapper of Menoh","items":[[3,"Buffer","menoh","Buffer, a safe wrapper of menoh buffer control scheme.",null,null],[3,"Model","","Main struct running inference.",null,null],[3,"ModelBuilder","","Builder of model.",null,null],[3,"ModelData","","Represent model data defined by ONNX file.",null,null],[3,"VariableProfile","","",null,null],[12,"dtype","","",0,null],[12,"dims","","",0,null],[3,"VariableProfileTable","","Variable Profile table.",null,null],[3,"VariableProfileTableBuilder","","Builder of Variable Profile Table.",null,null],[4,"Dtype","","Dtype that accepted by menoh model.",null,null],[13,"Float","","",1,null],[4,"Error","","",null,null],[13,"StdError","","",2,null],[13,"UnknownError","","",2,null],[13,"InvalidFileName","","",2,null],[13,"UnsupportedONNXOpsetVersion","","",2,null],[13,"ONNXParseError","","",2,null],[13,"InvalidDtype","","",2,null],[13,"InvalidAttributeType","","",2,null],[13,"UnsupportedOperatorAttribute","","",2,null],[13,"DimensionMismatch","","",2,null],[13,"VariableNotFound","","",2,null],[13,"IndexOutOfRange","","",2,null],[13,"JsonParseError","","",2,null],[13,"InvalidBackendName","","",2,null],[13,"UnsupportedOperator","","",2,null],[13,"FailedToConfigureOperator","","",2,null],[13,"BackendError","","",2,null],[13,"SameNamedVariableAlreadyExist","","",2,null],[13,"InvalidBufferSize","","",2,null],[13,"NotInternalBuffer","","",2,null],[0,"ffi","","This module provides \"raw\" declarations and linkage for `Menoh` C API.",null,null],[3,"menoh_model_data","menoh::ffi","",null,null],[3,"menoh_model","","",null,null],[3,"menoh_model_builder","","",null,null],[3,"menoh_variable_profile_table","","",null,null],[3,"menoh_variable_profile_table_builder","","",null,null],[5,"menoh_get_last_error_message","","",null,null],[5,"menoh_make_model_data_from_onnx","","",null,null],[5,"menoh_delete_model_data","","",null,null],[5,"menoh_model_data_optimize","","",null,null],[5,"menoh_make_variable_profile_table_builder","","",null,null],[5,"menoh_delete_variable_profile_table_builder","","",null,null],[5,"menoh_variable_profile_table_builder_add_input_profile_dims_2","","",null,null],[5,"menoh_variable_profile_table_builder_add_input_profile_dims_4","","",null,null],[5,"menoh_variable_profile_table_builder_add_output_profile","","",null,null],[5,"menoh_build_variable_profile_table","","",null,null],[5,"menoh_delete_variable_profile_table","","",null,null],[5,"menoh_variable_profile_table_get_dtype","","",null,null],[5,"menoh_variable_profile_table_get_dims_size","","",null,null],[5,"menoh_variable_profile_table_get_dims_at","","",null,null],[5,"menoh_make_model_builder","","",null,null],[5,"menoh_delete_model_builder","","",null,null],[5,"menoh_model_builder_attach_external_buffer","","",null,null],[5,"menoh_build_model","","",null,null],[5,"menoh_delete_model","","",null,null],[5,"menoh_model_get_variable_buffer_handle","","",null,null],[5,"menoh_model_get_variable_dtype","","",null,null],[5,"menoh_model_get_variable_dims_size","","",null,null],[5,"menoh_model_get_variable_dims_at","","",null,null],[5,"menoh_model_run","","",null,null],[6,"menoh_error_code","","",null,null],[6,"menoh_dtype","","",null,null],[6,"menoh_model_data_handle","","",null,null],[6,"menoh_model_handle","","",null,null],[6,"menoh_model_builder_handle","","",null,null],[6,"menoh_variable_profile_table_handle","","",null,null],[6,"menoh_variable_profile_table_builder_handle","","",null,null],[17,"menoh_error_code_success","","",null,null],[17,"menoh_error_code_std_error","","",null,null],[17,"menoh_error_code_unknown_error","","",null,null],[17,"menoh_error_code_invalid_filename","","",null,null],[17,"menoh_error_code_unsupported_onnx_opset_version","","",null,null],[17,"menoh_error_code_onnx_parse_error","","",null,null],[17,"menoh_error_code_invalid_dtype","","",null,null],[17,"menoh_error_code_invalid_attribute_type","","",null,null],[17,"menoh_error_code_unsupported_operator_attribute","","",null,null],[17,"menoh_error_code_dimension_mismatch","","",null,null],[17,"menoh_error_code_variable_not_found","","",null,null],[17,"menoh_error_code_index_out_of_range","","",null,null],[17,"menoh_error_code_json_parse_error","","",null,null],[17,"menoh_error_code_invalid_backend_name","","",null,null],[17,"menoh_error_code_unsupported_operator","","",null,null],[17,"menoh_error_code_failed_to_configure_operator","","",null,null],[17,"menoh_error_code_backend_error","","",null,null],[17,"menoh_error_code_same_named_variable_already_exist","","",null,null],[17,"menoh_dtype_float","","",null,null],[11,"new","menoh","Create buffer.",3,null],[11,"update","","Update buffer content from other data.",3,null],[11,"as_slice","","",3,null],[11,"clone","","",1,{"inputs":[{"name":"self"}],"output":{"name":"dtype"}}],[11,"value","","",1,{"inputs":[{"name":"self"}],"output":{"name":"menoh_dtype"}}],[11,"from","","",1,{"inputs":[{"name":"menoh_dtype"}],"output":{"name":"self"}}],[11,"type_id","","",1,{"inputs":[{"name":"self"}],"output":{"name":"typeid"}}],[11,"is_compatible","","",1,{"inputs":[{"name":"self"}],"output":{"name":"bool"}}],[11,"fmt","","",2,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"eq","","",2,{"inputs":[{"name":"self"},{"name":"error"}],"output":{"name":"bool"}}],[11,"fmt","","",2,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"new","","",4,{"inputs":[{"name":"variableprofiletable"}],"output":{"generics":["error"],"name":"result"}}],[11,"attach_external_buffer","","Attach external buffer to the [`Model`][Model] generated from this instance.",4,{"inputs":[{"name":"self"},{"name":"str"},{"name":"buffer"},{"name":"variableprofiletable"}],"output":{"generics":["error"],"name":"result"}}],[11,"attach_external_buffer_unchecked","","Attach external buffer to the [`Model`][Model] generated from this instance.",4,{"inputs":[{"name":"self"},{"name":"str"},{"name":"buffer"}],"output":{"generics":["error"],"name":"result"}}],[11,"build_model","","Build model.",4,{"inputs":[{"name":"self"},{"name":"modeldata"},{"name":"str"},{"name":"str"}],"output":{"generics":["model","error"],"name":"result"}}],[11,"run","","Run model inference",5,{"inputs":[{"name":"self"}],"output":{"generics":["error"],"name":"result"}}],[11,"get_attached_buffer","","Get attached buffer.",5,{"inputs":[{"name":"self"},{"name":"str"}],"output":{"generics":["buffer","error"],"name":"result"}}],[11,"get_internal_buffer","","Get reference to buffer generated inside model.",5,{"inputs":[{"name":"self"},{"name":"str"}],"output":{"generics":["buffer","error"],"name":"result"}}],[11,"get_variable_dtype","","Get dtype by name",5,{"inputs":[{"name":"self"},{"name":"str"}],"output":{"generics":["dtype","error"],"name":"result"}}],[11,"get_variable_dims","","Get dims by name",5,{"inputs":[{"name":"self"},{"name":"str"}],"output":{"generics":["vec","error"],"name":"result"}}],[11,"drop","","",4,{"inputs":[{"name":"self"}],"output":null}],[11,"drop","","",5,{"inputs":[{"name":"self"}],"output":null}],[11,"new","","Create ModelData from given ONNX file.",6,{"inputs":[{"name":"p"}],"output":{"generics":["error"],"name":"result"}}],[11,"optimize","","Optimize model data",6,{"inputs":[{"name":"self"},{"name":"variableprofiletable"}],"output":{"generics":["error"],"name":"result"}}],[11,"drop","","",6,{"inputs":[{"name":"self"}],"output":null}],[11,"get_variable_profile","","Get Variable profile detail by using variable name.",7,{"inputs":[{"name":"self"},{"name":"str"}],"output":{"generics":["variableprofile","error"],"name":"result"}}],[11,"new","","",8,{"inputs":[],"output":{"generics":["error"],"name":"result"}}],[11,"add_input_profile","","Add input profile. dims length must be 2 or 4.",8,null],[11,"add_output_profile","","Add output profile.",8,{"inputs":[{"name":"self"},{"name":"str"},{"name":"dtype"}],"output":{"generics":["error"],"name":"result"}}],[11,"build_variable_profile_table","","Build variable profile table.",8,{"inputs":[{"name":"self"},{"name":"modeldata"}],"output":{"generics":["variableprofiletable","error"],"name":"result"}}],[11,"drop","","",7,{"inputs":[{"name":"self"}],"output":null}],[11,"drop","","",8,{"inputs":[{"name":"self"}],"output":null}],[8,"DtypeCompatible","","Indicate compatible type with menoh dtype",null,null]],"paths":[[3,"VariableProfile"],[4,"Dtype"],[4,"Error"],[3,"Buffer"],[3,"ModelBuilder"],[3,"Model"],[3,"ModelData"],[3,"VariableProfileTable"],[3,"VariableProfileTableBuilder"]]};
initSearch(searchIndex);