"""
Лабораторная работа №1 — Продвинутый NumPy.

Реализация пяти задач:
  1. Числа с плавающей точкой (float32/float64, машинное эпсилон,
     катастрофическая отмена)
  2. Индексация, срезы и broadcasting
  3. Memory-mapped файлы и оптимизация кэша
  4. Разреженные матрицы (CSR / COO / CSC)
  5. Нативное расширение с Cython (сумма квадратов)

Запуск:
    pip install -r requirements.txt
    python setup.py build_ext --inplace   # скомпилировать Cython-модуль
    python lab1_numpy.py
"""

import timeit

import numpy as np
from scipy import sparse

SEP = "=" * 70


# ----------------------------------------------------------------------
# Задача 1. Числа с плавающей точкой
# ----------------------------------------------------------------------
def task1():
    print(SEP)
    print("ЗАДАЧА 1: Числа с плавающей точкой")
    print(SEP)

    # 1.1 Сумма float32 [0.1, 0.2, 0.3] в сравнении с точным значением 0.6
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    s = arr.sum()
    print("1.1 Массив float32 [0.1, 0.2, 0.3]:")
    print(f"    сумма (float32)   = {s!r}")
    print(f"    сумма (в float64) = {float(s):.17g}")
    print(f"    точное значение   = 0.6")
    print(f"    разница           = {abs(float(s) - 0.6):.2e}")
    print("    Объяснение: 0.1, 0.2 и 0.3 нельзя точно представить в двоичной")
    print("    системе счисления, поэтому при сложении накапливается ошибка")
    print("    округления.")

    # 1.2 Машинное эпсилон
    print("\n1.2 Машинное эпсилон (np.finfo):")
    print(f"    float32: {np.finfo(np.float32).eps:.2e}")
    print(f"    float64: {np.finfo(np.float64).eps:.2e}")

    # 1.3 Катастрофическая отмена: (1.0 + 1e-8) - 1.0
    print("\n1.3 Катастрофическая отмена (1.0 + 1e-8) - 1.0:")
    x32 = np.float32(1.0)
    d32 = np.float32(1e-8)
    print(f"    float32: {(x32 + d32 - x32)!r}  (ожидалось 1e-8 — точность потеряна)")

    x64 = np.float64(1.0)
    d64 = np.float64(1e-8)
    print(f"    float64: {(x64 + d64 - x64)!r}")
    print("    Объяснение: 1e-8 меньше машинного эпсилона float32 (~1.19e-7),")
    print("    поэтому 1.0 + 1e-8 округляется обратно до 1.0, и разность даёт 0.")


# ----------------------------------------------------------------------
# Задача 2. Индексация, срезы и broadcasting
# ----------------------------------------------------------------------
def task2():
    print(SEP)
    print("ЗАДАЧА 2: Индексация, срезы и broadcasting")
    print(SEP)

    A = np.arange(20).reshape(4, 5) - 10
    print("Исходный массив A (4×5):")
    print(A)

    # 2.1 Базовые срезы
    print("\n2.1 Базовые срезы:")
    print("    все элементы второй строки:", A[1, :])
    print("    столбец с индексом 3:", A[:, 3])
    print("    подмассив строк 1–3 и столбцов 2–4:")
    print(A[1:4, 2:5])

    # 2.2 Fancy-индексация
    print("\n2.2 Fancy-индексация (строки [0, 2, 3], столбцы [1, 4]):")
    print(A[np.ix_([0, 2, 3], [1, 4])])

    # 2.3 Логическая индексация
    A[A < 0] = 0
    print("\n2.3 После замены отрицательных элементов на 0:")
    print(A)

    # 2.4 Broadcasting
    B = np.ones((4, 1))
    C = np.full((1, 5), 2.0)
    print("\n2.4 Broadcasting: B(4, 1) + C(1, 5) =")
    print(B + C)
    print("    Объяснение: (4, 1) и (1, 5) расширяются по правилу broadcasting")
    print("    до общей формы (4, 5) без явных циклов.")


# ----------------------------------------------------------------------
# Задача 3. Memory-mapped файлы и оптимизация кэша
# ----------------------------------------------------------------------
def task3():
    print(SEP)
    print("ЗАДАЧА 3: Memory-mapped файлы и оптимизация кэша")
    print(SEP)

    filename = "mmap_array.dat"

    # 3.1 Memory-mapped файл для массива (1000, 1000) float64
    mm = np.memmap(filename, dtype="float64", mode="w+", shape=(1000, 1000))
    mm[:] = np.random.rand(1000, 1000)
    mm.flush()
    print("3.1 Создан memory-mapped файл для массива (1000, 1000) float64:")
    print(f"    файл: {filename}")
    print(f"    форма: {mm.shape}, dtype: {mm.dtype}")
    print(f"    размер на диске: {mm.nbytes / 1e6:.1f} МБ")

    # 3.2 Чтение отдельной строки и столбца (без загрузки всего массива)
    row = mm[0, :]
    col = mm[:, 0]
    print("\n3.2 Прочитаны отдельные строка и столбец (не весь массив):")
    print(f"    строка 0: форма {row.shape}, первые 5 значений: {row[:5]}")
    print(f"    столбец 0: форма {col.shape}, первые 5 значений: {col[:5]}")

    # 3.3 Сравнение времени обхода по строкам и по столбцам.
    # В интерактивной среде (Jupyter/IPython) здесь используется магия
    #     %timeit sum_row_major(mm)
    #     %timeit sum_col_major(mm)
    # В обычном .py-скрипте эквивалент — модуль timeit.
    def sum_row_major(m):
        total = 0.0
        for i in range(m.shape[0]):
            total += np.sum(m[i, :])   # строка лежит в памяти подряд
        return total

    def sum_col_major(m):
        total = 0.0
        for j in range(m.shape[1]):
            total += np.sum(m[:, j])   # столбец — элементы идут с большим шагом
        return total

    print("\n3.3 Время обхода массива (timeit.repeat, аналог %timeit):")
    t_row = min(timeit.repeat(lambda: sum_row_major(mm), number=1, repeat=5))
    t_col = min(timeit.repeat(lambda: sum_col_major(mm), number=1, repeat=5))
    print(f"    по строкам (row-major):   {t_row:.4f} с")
    print(f"    по столбцам (column-major): {t_col:.4f} с")
    print(f"    отношение: {t_col / t_row:.1f}x")
    print("    Объяснение: NumPy хранит массивы в порядке C (row-major) —")
    print("    элементы одной строки лежат в памяти подряд. При обходе по строкам")
    print("    процессор читает соседние адреса и данные попадают в кэш CPU")
    print("    (высокая локальность). При обходе по столбцам элементы разбросаны")
    print("    с шагом в целую строку, что даёт много промахов кэша и замедление.")


# ----------------------------------------------------------------------
# Задача 4. Разреженные матрицы
# ----------------------------------------------------------------------
def task4():
    print(SEP)
    print("ЗАДАЧА 4: Разреженные матрицы")
    print(SEP)

    # 4.1 Разреженная матрица (5, 5) с плотностью ~20% в формате CSR
    S = sparse.random(5, 5, density=0.2, format="csr", random_state=42)
    print("4.1 Разреженная матрица (5, 5), плотность ~20%, формат CSR:")
    print(f"    число ненулевых элементов: {S.nnz}")
    print("    плотный вид матрицы:")
    print(S.toarray())

    # 4.2 Преобразование в COO и CSC
    S_coo = S.tocoo()
    S_csc = S.tocsc()
    print("\n4.2 Преобразования:")
    print(f"    COO: {S_coo.nnz} ненулевых элементов")
    print(f"    CSC: {S_csc.nnz} ненулевых элементов")

    # 4.3 Умножение разреженной матрицы на плотный вектор
    v = np.ones(5)
    print("\n4.3 Умножение CSR-матрицы на плотный вектор (все единицы):")
    print("    ", S.dot(v))

    # 4.4 Сравнение скорости разреженной и плотной матриц (1000×1000)
    n = 1000
    S_big = sparse.random(n, n, density=0.01, format="csr", random_state=42)
    D_big = S_big.toarray()
    v_big = np.ones(n)

    t_sparse = min(timeit.repeat(lambda: S_big.dot(v_big), number=1, repeat=5))
    t_dense = min(timeit.repeat(lambda: D_big.dot(v_big), number=1, repeat=5))
    print("\n4.4 Умножение на вектор (1000×1000, плотность 0.01):")
    print(f"    разреженная (CSR): {t_sparse:.6f} с")
    print(f"    плотная:           {t_dense:.6f} с")
    print(f"    ускорение:         {t_dense / t_sparse:.1f}x")


# ----------------------------------------------------------------------
# Задача 5. Нативное расширение с Cython
# ----------------------------------------------------------------------
def task5():
    print(SEP)
    print("ЗАДАЧА 5: Нативное расширение с Cython")
    print(SEP)

    # 5.1 Функция на чистом Python: сумма квадратов элементов списка
    def sum_of_squares_py(xs):
        total = 0.0
        for x in xs:
            total += x * x
        return total

    print("5.1 Функция sum_of_squares_py (чистый Python, по списку) определена.")

    # 5.2 Cython-модуль (отдельный файл sum_of_squares_cy.pyx)
    try:
        import sum_of_squares_cy
    except ImportError:
        print("\n5.2 Модуль Cython НЕ скомпилирован.")
        print("    Скомпилируйте его командой: python setup.py build_ext --inplace")
        print("    и запустите скрипт ещё раз.")
        return

    print("5.2 Cython-модуль sum_of_squares_cy импортирован (скомпилирован из .pyx).")

    # 5.3 Сравнение времени на массиве из 10^7 элементов
    n = 10_000_000
    arr = np.random.rand(n)      # массив float64
    lst = arr.tolist()           # тот же набор значений как список

    py_res = sum_of_squares_py(lst)
    cy_res = sum_of_squares_cy.sum_of_squares_cy(arr)
    print(f"\n5.3 Корректность: Python={py_res:.10f}, Cython={cy_res:.10f}, "
          f"совпадают: {np.isclose(py_res, cy_res)}")

    t_py = min(timeit.repeat(lambda: sum_of_squares_py(lst), number=1, repeat=3))
    t_cy = min(timeit.repeat(lambda: sum_of_squares_cy.sum_of_squares_cy(arr),
                             number=1, repeat=3))

    print(f"    элементов: {n}")
    print(f"    Python (цикл по списку):      {t_py:.4f} с")
    print(f"    Cython (типизированный цикл): {t_cy:.4f} с")
    print(f"    ускорение: {t_py / t_cy:.1f}x")


def main():
    task1()
    print()
    task2()
    print()
    task3()
    print()
    task4()
    print()
    task5()
    print()
    print(SEP)
    print("Лабораторная работа выполнена.")


if __name__ == "__main__":
    main()
