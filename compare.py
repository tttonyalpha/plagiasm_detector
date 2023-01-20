import ast
import argparse


class Vst(ast.NodeVisitor):

    def __init__(self):
        self.dict = {'names': [], 'operators': [],
                     'constants': [], 'imports': [], 'structure': []}

    def visit(self, node: ast.AST):
        # из-за мелого количества времени я вынужден ограничиться такими признаками, однако,
        # есть еще множество идей для других признаков:

        # имена переменных функций и т.д. жа схожесть жестко баним
        if isinstance(node, ast.Name):
            self.dict['names'].append(node.id)
        # бинарные операции
        elif isinstance(node, ast.BinOp):
            self.dict['operators'].append(node.op)
        # числовые константы
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, str):
                self.dict['constants'].append(node.value)
        # ипорты; можно и нужно еще учесть ипорты в виде from xxx import yyy, но опять же времени мало, а реализовать это понятно как
        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.dict['imports'].append(alias.name)
        # print(type(node).__name__)

        # общая структура программы
        if (isinstance(node, ast.stmt) or isinstance(node, ast.expr)) and not isinstance(node, ast.Expr):
            self.dict['structure'].append(type(node).__name__)

        self.generic_visit(node)

    def repr(self):
        return self.dict

# википедия; считаем что цены за вставку, удаление, подстановку и транспозицию одинаковые


def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    if lenstr1 == 0 and lenstr2 == 0:
        return 1
    elif lenstr1 == 0 or lenstr2 == 0:
        return 0
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # удаление
                d[(i, j - 1)] + 1,  # вставка
                d[(i - 1, j - 1)] + cost,  # подстановка
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + 1)  # транспозиция

    return 1 - d[lenstr1 - 1, lenstr2 - 1] / max(lenstr1, lenstr2)


def main():

    parser = argparse.ArgumentParser(description='argparser')
    parser.add_argument('input_file', type=str, help='input')
    parser.add_argument('output_file', type=str, help='output')
    args = parser.parse_args()

    foutput = open(args.output_file, 'w')

    with open(args.input_file) as finput:
        for line in finput:
            marked_score, path_1, path_2 = line.strip().split()
            print(path_1, path_2)
            with open(path_1) as f1, open(path_2) as f2:
                code_1 = f1.read()
                code_2 = f2.read()
                try:
                    tree = ast.parse(code_1)
                except:
                    continue
                visitor = Vst()
                visitor.visit(tree)
                repr_1 = visitor.repr()
                try:
                    tree = ast.parse(code_2)
                except:
                    continue
                visitor = Vst()
                visitor.visit(tree)
                repr_2 = visitor.repr()

                names_score = damerau_levenshtein_distance(
                    sorted(repr_1['names']), sorted(repr_2['names']))
                operators_score = damerau_levenshtein_distance(
                    repr_1['operators'], repr_2['operators'])
                constants_score = damerau_levenshtein_distance(
                    repr_1['constants'], repr_2['constants'])
                imports_score = damerau_levenshtein_distance(
                    sorted(repr_1['imports']), sorted(repr_2['imports']))
                structure_score = damerau_levenshtein_distance(
                    sorted(repr_1['structure']), sorted(repr_2['structure']))

                # по факту здесь мы обучим модельку и веса будем получать линейной регрессией, но я не успеваю, поэтому беру веса из соображения логики
                # и понятно, что веса, даже если их брать из головы, должны подстраиваться под задачу, например, мы хотим проверить плагиат домашек,
                # где преподаватель заранее прописал все имена функций и переменных, тогда очевидно, что вес для names_score нужно уменьшит
                w_1 = 0.4
                w_2 = 0.1
                w_3 = 0.1
                #w_4 = 0.05
                w_5 = 0.35

                # print(names_score, operators_score, constants_score,
                #      imports_score, structure_score)
                final_score = (names_score, operators_score,
                               constants_score, structure_score)
                # print(final_score)
                foutput.write(
                    f"{marked_score} {names_score} {operators_score} {constants_score} {imports_score} {structure_score}\n")

    foutput.close()


if __name__ == '__main__':
    main()
