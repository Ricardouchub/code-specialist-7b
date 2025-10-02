# benchmark_tasks.py
TASKS = [
    # Más fáciles que tu set original y con énfasis en obedecer “solo el cuerpo”
    ("add", 
     "def add(a, b):\n    \"\"\"Devuelve a + b\"\"\"\n    # Escribe solo el cuerpo de la función, sin cambiar la firma.\n    # Prohibido: imports, prints, explicaciones.\n",
     "import pytest\n\ndef test_add():\n    assert add(1,2)==3\n    assert add(-1,1)==0\n"),

    ("is_even", 
     "def is_even(n:int)->bool:\n    \"\"\"True si n es par\"\"\"\n    # Solo el cuerpo, no cambies la firma ni el nombre.\n",
     "import pytest\n\ndef test_e():\n    assert is_even(2)\n    assert not is_even(3)\n"),

    ("reverse_words", 
     "def reverse_words(s:str)->str:\n    \"\"\"Invierte el orden de las palabras separadas por espacios\"\"\"\n    # Solo el cuerpo, sin imports ni prints.\n",
     "import pytest\n\ndef test_rev():\n    assert reverse_words('hola mundo')=='mundo hola'\n    assert reverse_words('a b c')=='c b a'\n"),

    ("count_vowels", 
     "def count_vowels(s:str)->int:\n    \"\"\"Cuenta vocales minúsculas aeiou\"\"\"\n    # Solo el cuerpo, evita bucles innecesarios.\n",
     "import pytest\n\ndef test_v():\n    assert count_vowels('hola')==2\n    assert count_vowels('xyz')==0\n"),

    ("unique_sorted", 
     "def unique_sorted(lst:list)->list:\n    \"\"\"Devuelve elementos únicos ordenados\"\"\"\n    # Solo el cuerpo; no cambies el tipo de retorno.\n",
     "import pytest\n\ndef test_u():\n    assert unique_sorted([3,1,2,3,2,1])==[1,2,3]\n"),

    ("sum_digits", 
     "def sum_digits(n:int)->int:\n    \"\"\"Suma dígitos de un entero no negativo\"\"\"\n    # Solo el cuerpo; no uses print.\n",
     "import pytest\n\ndef test_s():\n    assert sum_digits(0)==0\n    assert sum_digits(12345)==15\n"),

    ("is_palindrome", 
     "def is_palindrome(s:str)->bool:\n    \"\"\"Palíndromo ignorando espacios y mayúsculas\"\"\"\n    # Solo el cuerpo.\n",
     "import pytest\n\ndef test_pal():\n    assert is_palindrome('Anita lava la tina')\n    assert not is_palindrome('python')\n"),

    ("fact", 
     "def fact(n:int)->int:\n    \"\"\"n! para n>=0\"\"\"\n    # Solo el cuerpo; maneja casos base correctamente.\n",
     "import pytest\n\ndef test_fact():\n    assert fact(0)==1\n    assert fact(5)==120\n"),

    ("unique_chars", 
     "def unique_chars(s:str)->bool:\n    \"\"\"True si todos los caracteres son únicos\"\"\"\n    # Solo el cuerpo; evita prints.\n",
     "import pytest\n\ndef test_uc():\n    assert unique_chars('abc')\n    assert not unique_chars('aba')\n"),

    ("median", 
     "def median(nums:list)->float:\n    \"\"\"Mediana de una lista no vacía\"\"\"\n    # Solo el cuerpo; no uses librerías externas.\n",
     "import pytest\n\ndef test_med():\n    assert median([1,3,2])==2\n    assert median([1,2,3,4])==2.5\n"),
]