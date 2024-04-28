; ModuleID = 'vectorAddModule'
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

; Function Attrs: convergent noinline nounwind
define void @vectorAdd(float* %A, float* %B, float* %C, i32 %N) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bdim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block_offset = mul i32 %bid, %bdim
  %idx = add i32 %block_offset, %tid

  %cmp = icmp slt i32 %idx, %N
    br i1 %cmp, label %if.then, label %if.end

  if.then:                                         ; preds = %entry
    %arrayidx = getelementptr inbounds float, float* %A, i32 %idx
    %0 = load float, float* %arrayidx, align 4
    %arrayidx1 = getelementptr inbounds float, float* %B, i32 %idx
    %1 = load float, float* %arrayidx1, align 4
    %add = fadd float %0, %1
    %arrayidx2 = getelementptr inbounds float, float* %C, i32 %idx
    store float %add, float* %arrayidx2, align 4
    br label %if.end

  if.end:                                          ; preds = %if.then, %entry
    ret void
  }

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

; Annotations for marking the kernel
!nvvm.annotations = !{!0}
!0 = !{void (float*, float*, float*, i32)* @vectorAdd, !"kernel", i32 1}
