file(REMOVE_RECURSE
  "../lib/libtvm_ffi_static.a"
  "../lib/libtvm_ffi_static.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/tvm_ffi_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
