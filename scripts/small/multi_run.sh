for MODEL in gp bnn llm; do
    for DATA in ESOL; do
        for _ in {1..10}; do
            MODEL=$MODEL DATA=$DATA bsub < run.sh;
        done
    done
done