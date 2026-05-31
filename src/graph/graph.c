#include "../../inc/graph/graph.h"

#include <assert.h>
#include <stdlib.h>
Graph graph_init(Context *ctx, size_t n_nodes){
    Graph graph;
    graph.ctx = ctx;
    graph.nodes = calloc(n_nodes, sizeof(Tensor));
    graph.curr_node = 0;
    graph.n_nodes = n_nodes;
    return graph;
}

void graph_free(Graph *graph){
    assert(graph->n_nodes > 0);
    free(graph->nodes);
    graph->n_nodes = 0;
}

Tensor *graph_alloc_node(Graph *graph){
    assert(graph->curr_node <= graph->n_nodes);
    Tensor *node = &graph->nodes[graph->curr_node];
    graph->curr_node++;
    return node;
}


void graph_print(const Graph *graph){
    tensor_print_header();
    for(size_t i = 0; i < graph->n_nodes; i++){
        tensor_print(&graph->nodes[i]);
    }
}

#include <stdio.h>

void graph_export_dot(const Graph *graph, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return;
    }

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "    rankdir=LR;\n");
    fprintf(fp, "    node [shape=box, style=rounded];\n\n");

    for (size_t i = 0; i < graph->n_nodes; i++) {
        const Tensor *t = &graph->nodes[i];

        fprintf(fp,
                "    n%zu [label=\"%s\\n%s\"];\n",
                t->id,
                t->name,
                op_name(t->op_type));
    }

    fprintf(fp, "\n");

    for (size_t i = 0; i < graph->n_nodes; i++) {
        const Tensor *t = &graph->nodes[i];

        for (size_t j = 0; j < t->nsrc; j++) {
            fprintf(fp,
                    "    n%zu -> n%zu;\n",
                    t->src[j]->id,
                    t->id);
        }
    }

    fprintf(fp, "}\n");
    fclose(fp);
}



