# Dynamic-VM-allocation

Добавлены 4 основные эвристики (next-fit, first-fit, best-fit, worst-fit). Доступна слеующая симуляция: генерируется 1000 реальных и 2000 виртуальных машин, виртуальные машины помещаются на серверы при помощи одной из эвристик (без миграций, по одной), считается потребление энергии, время работы, число задействованных хостов и т.д.

#### ========== 22.11.2020 ==========
`generator.py` - генерирует .json файл с параметрами симуляции

`simulator.py` - собственно, сама симуляция

`python 3 generator.py` - сгенерировать синтетические данные

Запустить симуляцию можно так:
```
  >>> python3 simulator.py next-fit 
  Total time: 0.006s
  Total cumulative energy: 4.505^+06
  Total hosts active: 857
  Total VMs allocated: 2000
  CPU global utilization: 0.67449
  RAM global utilization: 0.67566
```
Ещё есть варианты `first-fit`, `best-fit`, `worst-fit`
