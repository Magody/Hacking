with open("helper.txt") as f:
    lines = f.readlines()
    out = "["
    for line in lines:
        out += f"\"{line.split(' =')[0]}\","
    out = out[:-1] + "]"
    print(out)