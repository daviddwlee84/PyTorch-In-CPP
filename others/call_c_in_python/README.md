# Call C Library in Python

```bash
gcc division.c
./a.out

# Now create a shared library
gcc -fPIC -shared -o libdiv.so division.c
```

- [Passing a callback function from Python to C | by Prince Francis | Medium](https://princekfrancis.medium.com/passing-a-callback-function-from-python-to-c-351ac944e041)
