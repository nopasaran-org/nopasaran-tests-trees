import pydot
import random
from PIL import Image, PngImagePlugin
from io import BytesIO

class TestsTreeNode:
    def __init__(self, name, inputs=None, outputs=None, test=None, default_input_values=None):
        self.name = name
        self.inputs = inputs if inputs else []
        self.outputs = outputs if outputs else []
        self.test = test
        self.children = []
        self.default_input_values = default_input_values if default_input_values else {}

    def add_child(self, child, conditions):
        self.children.append((child, conditions))



    def evaluate_test(self, endpoints, repository, input_values):
        result = [str(random.randint(1, 20)) for _ in self.outputs]  # Dummy evaluation for the example
        return result
        
class TestsTree:
    def __init__(self, repository=None, endpoints=None):
        self.root = None
        self.repository = repository
        self.endpoints = endpoints if endpoints else []

    def add_root(self, node):
        self.root = node

    def add_edge(self, parent, child, conditions):
        parent.add_child(child, conditions)
    
    def to_dot(self):
        graph = pydot.Dot(graph_type='digraph', bgcolor='white')

        if self.repository:
            repository_label = f'Repository: {self.repository}'
            graph.set_label(repository_label)
            graph.set_labelloc('t')  # Place the label at the top
            graph.set_labeljust('l')  # Left-justify the label

        node_style = {
            'fontname': 'Arial',
            'fontsize': '10',
            'style': 'filled',
            'fillcolor': '#f0f0f0',  # Light gray fill color
            'color': '#666666',  # Border color
        }
        edge_style = {
            'fontsize': '9',
            'fontcolor': '#333333',  # Edge label font color
            'color': '#999999',  # Edge color
        }

        self._add_nodes_edges(self.root, graph, node_style=node_style, edge_style=edge_style)

        return graph

    def from_dot(self, dot_string):
        graph = pydot.graph_from_dot_data(dot_string)[0]
        nodes = {}

        # Extract repository from graph label
        repository_label = graph.get_label()
        if repository_label:
            self.repository = repository_label.replace('Repository: ', '').strip()

        for node in graph.get_nodes():
            name = node.get_name().strip('"')
            inputs = []
            outputs = []
            test = None
            default_input_values = {}

            # Parse the label to extract inputs, outputs, and test
            label = node.get_attributes()['label'].strip('"')
            label_parts = label.split('\\n')
            main_label = label_parts[0]

            for part in label_parts[1:]:
                if part.startswith('Inputs: ['):
                    inputs_str = part.replace('Inputs: [', '').replace(']', '')
                    inputs = [inp.split('=')[0].strip() for inp in inputs_str.split(',')]
                    default_input_values = {inp.split('=')[0].strip(): int(inp.split('=')[1].strip()) for inp in inputs_str.split(',') if '=' in inp}
                elif part.startswith('Outputs: '):
                    outputs_str = part.replace('Outputs: ', '').replace("[", "").replace("]", "")
                    outputs = outputs_str.split(', ')
                elif part.startswith('Test: '):
                    test = part.replace('Test: ', '').strip()

            nodes[name] = TestsTreeNode(name, inputs, outputs, test, default_input_values)

        for edge in graph.get_edges():
            parent_name = edge.get_source().strip('"')
            child_name = edge.get_destination().strip('"')
            conditions = [edge.get_attributes()['label'].strip('"')]
            self.add_edge(nodes[parent_name], nodes[child_name], conditions)

        if nodes:
            self.root = nodes[graph.get_node_list()[0].get_name().strip('"')]

        return self

    def _add_nodes_edges(self, node, graph, parent_default_inputs=None, node_style=None, edge_style=None):
        if node is None:
            return
        
        if parent_default_inputs:
            node_default_inputs = {**parent_default_inputs, **node.default_input_values}
        else:
            node_default_inputs = node.default_input_values
        
        formatted_inputs = self._format_inputs(node.inputs, node_default_inputs)
        default_inputs_str = ', '.join(formatted_inputs) if formatted_inputs else ''
        
        # Use node's name instead of label for node_label
        node_label = f'{node.name}'
        if default_inputs_str:
            node_label += f'\nInputs: [{default_inputs_str}]'
        if node.outputs:
            node_label += f'\nOutputs: [{", ".join(node.outputs)}]'
        if node.test:
            node_label += f'\nTest: {node.test}'
        
        node_attributes = {
            'label': node_label,
            'shape': 'box'
        }

        if node_style:
            node_attributes.update(node_style)

        graph_node = pydot.Node(node.name, **node_attributes)
        graph.add_node(graph_node)
        
        for child, conditions in node.children:
            self._add_nodes_edges(child, graph, node_default_inputs, node_style, edge_style)
            for condition in conditions:
                edge_attributes = {
                    'label': condition,
                    'fontsize': '9',
                    'fontcolor': '#333333',  # Edge label font color
                    'color': '#999999'  # Edge color
                }
                if edge_style:
                    edge_attributes.update(edge_style)
                graph.add_edge(pydot.Edge(node.name, child.name, **edge_attributes))

    def _format_inputs(self, inputs, default_inputs):
        formatted_inputs = []
        for inp in inputs:
            if inp in default_inputs:
                formatted_inputs.append(f'{inp}={default_inputs[inp]}')
            else:
                formatted_inputs.append(inp)
        return formatted_inputs

    def evaluate_tree(self, node_inputs=None):
        if node_inputs is None:
            node_inputs = {}
        return self._evaluate_node(self.root, node_inputs)

    def _evaluate_node(self, node, node_inputs):
        if node.name in node_inputs:
            merged_input_values = {**node.default_input_values, **node_inputs[node.name]}
        else:
            merged_input_values = node.default_input_values

        output_values = node.evaluate_test(self.endpoints, self.repository, merged_input_values)
        for idx, output_name in enumerate(node.outputs):
            merged_input_values[output_name] = output_values[idx]

        if node.children:
            for child, conditions in node.children:
                condition_evaluated = False
                for condition in conditions:
                    try:
                        if eval(condition, {}, merged_input_values):  # Use eval with a safe context
                            output_values = self._evaluate_node(child, node_inputs)
                            condition_evaluated = True
                            break
                    except Exception as e:
                        print(f"Error evaluating condition '{condition}' for node {node.name}: {e}")

                if condition_evaluated:
                    break

        return output_values

    def save_png_with_metadata(self, filename):
        graph = self.to_dot()
        png_str = graph.create_png()
        image = Image.open(BytesIO(png_str))

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("TestsTree", graph.to_string())
        metadata.add_text("Repository", self.repository)

        image.save(filename, "PNG", pnginfo=metadata)

    def load_from_png(self, filename):
        image = Image.open(filename)
        metadata = image.info.get("TestsTree")
        repository = image.info.get("Repository")
        if metadata:
            self.from_dot(metadata)
        if repository:
            self.repository = repository

    def load_from_png_content(self, png_content):
        with BytesIO(png_content) as file:
            image = Image.open(file)
            metadata = image.info.get("TestsTree")
            repository = image.info.get("Repository")
            if metadata:
                self.from_dot(metadata)
            if repository:
                self.repository = repository

class TestsTreeTest:
    @staticmethod
    def save_tree():
        root_default_inputs = {'X': 100}
        root = TestsTreeNode('Root', inputs=['X', 'Y'], outputs=['Z'], test='Example', default_input_values=root_default_inputs)

        child1_default_inputs = {'U': 5}
        child1 = TestsTreeNode('Child 1', inputs=['U'], outputs=['V'], test='Example', default_input_values=child1_default_inputs)

        child2_default_inputs = {'U': 10}
        child2 = TestsTreeNode('Child 2', inputs=['U'], outputs=['V'], test='Example', default_input_values=child2_default_inputs)

        repository = "https://github.com/dummy-repository-url"

        tree = TestsTree(repository=repository)
        tree.add_root(root)
        tree.add_edge(root, child1, ['Z < 10'])
        tree.add_edge(root, child2, ['Z >= 10'])

        png_filename = 'tests_tree.png'
        tree.save_png_with_metadata(png_filename)
        print(f"Tests tree image saved as '{png_filename}'.")

    @staticmethod
    def load_and_evaluate_tree():
        tree = TestsTree()
        png_filename = 'tests_tree.png'
        tree.load_from_png(png_filename)

        print(tree.repository)

        node_inputs = {
            'A': {'X': 10, 'Y': 6},
            'B': {'Z': 7},
            'C': {'Z': 15}
        }

        final_output = tree.evaluate_tree(node_inputs)
        print(f"Final output value: {final_output}")

# Run the tests
TestsTreeTest.save_tree()
TestsTreeTest.load_and_evaluate_tree()
