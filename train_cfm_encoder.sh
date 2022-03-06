mkdir -p out

mkdir -p out/cfm
mkdir -p out/cfm/torus
mkdir -p out/cfm/writer
mkdir -p out/cfm/rollingpin
mkdir -p out/cfm/pinch

mkdir -p out/inverse
mkdir -p out/inverse/torus
mkdir -p out/inverse/writer
mkdir -p out/inverse/rollingpin
mkdir -p out/inverse/pinch

mkdir -p out/forward
mkdir -p out/forward/torus
mkdir -p out/forward/writer
mkdir -p out/forward/rollingpin
mkdir -p out/forward/pinch

mkdir -p out/e2c
mkdir -p out/e2c/torus
mkdir -p out/e2c/writer
mkdir -p out/e2c/rollingpin
mkdir -p out/e2c/pinch

mkdir -p pretrain_model/inverse
mkdir -p pretrain_model/inverse/torus
mkdir -p pretrain_model/inverse/writer
mkdir -p pretrain_model/inverse/rollingpin
mkdir -p pretrain_model/inverse/pinch
mkdir -p pretrain_model/forward
mkdir -p pretrain_model/forward/torus
mkdir -p pretrain_model/forward/writer
mkdir -p pretrain_model/forward/rollingpin
mkdir -p pretrain_model/forward/pinch
mkdir -p pretrain_model/e2c
mkdir -p pretrain_model/e2c/torus
mkdir -p pretrain_model/e2c/writer
mkdir -p pretrain_model/e2c/rollingpin
mkdir -p pretrain_model/e2c/pinch

# python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l Forward &>out/forward/torus/20220306.out
# python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l Inverse &>out/inverse/torus/20220306.out
# python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l E2C --encoder_path pretrain_model/e2c/pretrain_encoders/torus.pth &>out/e2c/torus/20220306.out

# python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l Forward &>out/forward/writer/20220306.out
# python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l Inverse &>out/inverse/writer/20220306.out
# python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l E2C --encoder_path pretrain_model/e2c/pretrain_encoders/writer.pth &>out/e2c/writer/20220306.out

# python -m plb.algorithms.cfm.train_cfm -d rollingpin -b 32 -l Forward &>out/forward/rollingpin/20220306.out
# python -m plb.algorithms.cfm.train_cfm -d rollingpin -b 32 -l Inverse &>out/inverse/rollingpin/20220306.out
python -m plb.algorithms.cfm.train_cfm -d rollingpin -b 32 -l E2C --encoder_path pretrain_model/e2c/pretrain_encoders/rollingpin-v1.pth &>out/e2c/rollingpin/20220306.out

# python -m plb.algorithms.cfm.train_cfm -d pinch -b 32 -l Forward &>out/forward/pinch/20220306.out
# python -m plb.algorithms.cfm.train_cfm -d pinch -b 32 -l Inverse &>out/inverse/pinch/20220306.out
python -m plb.algorithms.cfm.train_cfm -d pinch -b 32 -l E2C --encoder_path pretrain_model/e2c/pretrain_encoders/pinch-v1.pth &>out/e2c/pinch/20220306.out

