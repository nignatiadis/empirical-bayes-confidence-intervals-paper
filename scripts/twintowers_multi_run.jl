for r in 1:10
    submit = `srun -t 10:00:00 julia twintowers_multi.jl $r`
    run(submit)
end