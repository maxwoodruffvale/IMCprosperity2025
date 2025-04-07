from collections import deque
q = deque()
exhange_rates = {"snowballs":[1,1.45,.52,.72],
                 "pizza":[.7,1,.31,.48],
                 "nuggets":[1.95,3.1,1,1.49],
                 "shells":[1.34,1.98,.64,1]}
names=["snowballs","pizza","nuggets","shells"]
q.append((500,"shells"))
x = 0
while x < 4:
    for j in range(len(q)):
        element = q.popleft()
        words = element[1].split()
        for i in range(4):
            q.append((element[0] * exhange_rates[words[-1]][i], element[1] + " " + names[i]))
    x += 1
greatest = 500
word = ""
for element in list(q):
    words = element[1].split()
    final = element[0] * exhange_rates[words[-1]][3]
    print(final)
    if final >= greatest:
        greatest = final
        word = element[1] + " " + names[3]
print(greatest,word)
