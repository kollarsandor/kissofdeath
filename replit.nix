{pkgs}: {  
  deps = [  
    pkgs.zig  
    pkgs.python3  
    pkgs.python3Packages.pip  
    pkgs.futhark  
    pkgs.z3  
    pkgs.llvm_17  
    pkgs.clang_17  
    pkgs.lld_17  
    pkgs.gcc  
    pkgs.gnumake  
    pkgs.curl  
    pkgs.gnutar  
    pkgs.gzip  
    pkgs.unzip  
    pkgs.yosys  
    pkgs.nextpnr  
    pkgs.icestorm  
    pkgs.verilator  
    pkgs.nodejs  
    pkgs.bash  
    pkgs.pkg-config  
  ];  
}
