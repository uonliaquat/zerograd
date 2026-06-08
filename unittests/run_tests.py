import subprocess

tests = [
    "unittests/test_add.py",
    "unittests/test_arange.py",
    "unittests/test_index.py",
    "unittests/test_layernorm.py"
]

passed = 0
failed = 0
for test in tests:
    result = subprocess.run(["python", test])
    if result.returncode == 0:
        passed += 1
    else:
        failed += 1



print()
print("=" * 50)
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Total : {passed + failed}")
print("=" * 50)
