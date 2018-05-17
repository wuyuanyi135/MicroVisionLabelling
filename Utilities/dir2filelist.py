import sys

print("input ls output then EOF (Ctrl+D in *nix Ctrl+Z in Windows)")

input_str = sys.stdin.read()

lines = input_str.split('\n')
outstring = ""
for line in lines:
    sections  = line.split(' ')
    outstring = outstring + "\"" +sections[-1]+ "\" "

print(outstring)




