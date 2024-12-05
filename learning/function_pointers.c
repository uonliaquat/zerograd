#include <stdio.h>
#include <stdlib.h>

// Function prototypes
void add_operation(float a, float b);
void subtract_operation(float a, float b);
void multiply_operation(float a, float b);

// Define a type for operation functions
typedef void (*OperationFunc)(float, float);

// Node structure representing an operation in a computational graph
typedef struct Node {
    int id;                     // Node ID
    float input1;               // First input
    float input2;               // Second input
    OperationFunc operation;    // Function pointer to the operation
} Node;

// Implementation of different operations
void add_operation(float a, float b) {
    printf("Add: %.2f + %.2f = %.2f\n", a, b, a + b);
}

void subtract_operation(float a, float b) {
    printf("Subtract: %.2f - %.2f = %.2f\n", a, b, a - b);
}

void multiply_operation(float a, float b) {
    printf("Multiply: %.2f * %.2f = %.2f\n", a, b, a * b);
}

// Function to execute the operation of a node
void execute_node(Node* node) {
    if (node->operation != NULL) {
        printf("Executing Node %d:\n", node->id);
        node->operation(node->input1, node->input2);
    } else {
        printf("Node %d has no operation assigned!\n", node->id);
    }
}

// Main function
int main() {
    // Define nodes
    Node node1 = {1, 10.0, 5.0, add_operation};
    Node node2 = {2, 10.0, 5.0, subtract_operation};
    Node node3 = {3, 10.0, 5.0, multiply_operation};
    Node node4 = {4, 10.0, 5.0, NULL};  // No operation assigned

    // Array of nodes
    Node* nodes[] = {&node1, &node2, &node3, &node4};
    int num_nodes = sizeof(nodes) / sizeof(nodes[0]);

    // Execute all nodes
    for (int i = 0; i < num_nodes; ++i) {
        execute_node(nodes[i]);
        printf("\n");
    }

    // Dynamic assignment of operations
    printf("Dynamically assigning multiply_operation to Node 4:\n");
    nodes[3]->operation = multiply_operation;
    execute_node(nodes[3]);

    return 0;
}

