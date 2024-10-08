import ast
import astor
from pathlib import Path
import re

class JoinToPathTransformer(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'os' and node.func.attr == 'path':
                if isinstance(node.func.ctx, ast.Load) and len(node.args) > 1:
                    new_node = node.args[0]
                    for arg in node.args[1:]:
                        new_node = ast.BinOp(left=new_node, op=ast.Div(), right=arg)
                    return new_node
        return node

def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    tree = ast.parse(content)
    transformer = JoinToPathTransformer()
    modified_tree = transformer.visit(tree)

    modified_content = astor.to_source(modified_tree)

    # Replace os.path.dirname with Path.parent
    modified_content = re.sub(r'os\.path\.dirname\((.*?)\)', r'Path(\1).parent', modified_content)

    if content != modified_content:
        with open(file_path, 'w') as file:
            file.write(modified_content)
        print(f"Modified: {file_path}")

def process_directory(directory):
    for path in Path(directory).rglob('*.py'):
        process_file(path)

if __name__ == "__main__":
    nilearn_path = Path("nilearn")  # Adjust this path if needed
    if nilearn_path.is_dir():
        process_directory(nilearn_path)
    else:
        print(f"Error: {nilearn_path} is not a directory or doesn't exist.")