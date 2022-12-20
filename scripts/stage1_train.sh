
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# python -m src.train --config=configs/stage1-effb1-f0.yml
# python -m src.train --config=configs/stage1-effb1-f1.yml
# python -m src.train --config=configs/stage1-effb1-f2.yml
# python -m src.train --config=configs/stage1-effb1-f3.yml
# python -m src.train --config=configs/stage1-effb1-f4.yml

python -m src.train --config=configs/stage1-rns14-f0.yml
python -m src.train --config=configs/stage1-rns14-f1.yml
python -m src.train --config=configs/stage1-rns14-f2.yml
python -m src.train --config=configs/stage1-rns14-f3.yml
python -m src.train --config=configs/stage1-rns14-f4.yml
