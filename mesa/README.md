# LIST Methods and MESA

## Bash Script for Testing Ease
```bash
CURR_DIR=$(pwd)
cd "[put your psi4 location]/psi4" || exit
export PATH=[put your psi4 location]/psi4/objdir_p4dev/stage/bin:$PATH
export PYTHONPATH=[put your psi4 location]/psi4/objdir_p4dev/stage/lib:$PYTHONPATH
eval "$(conda/psi4-path-advisor.py cmake)"
if [ "$1" = "--test" ]; then
  psi4 --test
fi
cd "$CURR_DIR" || exit
python loop_runner.py
```
