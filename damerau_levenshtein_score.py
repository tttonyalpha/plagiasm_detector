import ast


class Vst(ast.NodeVisitor):

    def __init__(self):
        self.dict = {'names': [], 'operators': [],
                     'constants': [], 'imports': [], 'structure': []}

    def visit(self, node: ast.AST):
        if isinstance(node, ast.Name):
            self.dict['names'].append(node.id)
        elif isinstance(node, ast.BinOp):
            self.dict['operators'].append(node.op)

        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, str):
                self.dict['constants'].append(node.value)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.dict['imports'].append(alias.name)

        if isinstance(node, (ast.stmt, ast.operator, ast.expr)) and not isinstance(node, ast.Expr):
            self.dict['structure'].append(type(node).__name__)

        self.generic_visit(node)

    def repr(self):
        return self.dict


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


def get_score(path_1, path_2):
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
        repr_2 = visitor.repr(
        names_score=damerau_levenshtein_distance(
            sorted(repr_1['names']), sorted(repr_2['names']))
        operators_score=damerau_levenshtein_distance(
            repr_1['operators'], repr_2['operators'])
        constants_score=damerau_levenshtein_distance(
            repr_1['constants'], repr_2['constants'])
        imports_score=damerau_levenshtein_distance(
            sorted(repr_1['imports']), sorted(repr_2['imports']))
        structure_score=damerau_levenshtein_distance(
            repr_1['structure'], repr_2['structure'])

    return (names_score, operators_score, constants_score, imports_score, structure_score)
