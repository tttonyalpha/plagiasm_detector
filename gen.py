names = []
with open('inpt.txt') as fin, open('input.txt', 'w') as fout:
    for line in fin.readlines():
        names.append(line.strip())

    for i in range(len(names)):
        fout.write('1 files/' + names[i] + ' ' + 'plagiat1/' + names[i] + '\n')
        fout.write('1 files/' + names[i] + ' ' + 'plagiat2/' + names[i] + '\n')
        if i + 1 < len(names):
            fout.write('0 files/' + names[i] + ' ' +
                       'plagiat1/' + names[i + 1] + '\n')
        else:
            fout.write('0 files/' + names[i] + ' ' +
                       'plagiat1/' + names[i - 1] + '\n')

        if i + 1 < len(names):
            fout.write('0 files/' + names[i] + ' ' +
                       'plagiat2/' + names[i + 1] + '\n')
        else:
            fout.write('0 files/' + names[i] + ' ' +
                       'plagiat2/' + names[i - 1] + '\n')
