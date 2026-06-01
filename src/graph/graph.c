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

void graph_plan_memory(Graph *graph){

}


void graph_print(const Graph *graph){
    tensor_print_header();
    for(size_t i = 0; i < graph->n_nodes; i++){
        tensor_print(&graph->nodes[i]);
    }
}
#include <stdio.h>
#include <string.h>

void graph_export_dot(const Graph *graph, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return;
    }

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "    rankdir=TB;\n");
    fprintf(fp, "    nodesep=0.35;\n");
    fprintf(fp, "    ranksep=0.7;\n");
    fprintf(fp, "    splines=ortho;\n");
    fprintf(fp, "    bgcolor=\"white\";\n\n");

    fprintf(fp,
        "    node [\n"
        "        shape=plaintext,\n"
        "        margin=0\n"
        "    ];\n\n");

    fprintf(fp,
        "    edge [\n"
        "        color=\"#94A3B8\",\n"
        "        penwidth=1.5,\n"
        "        arrowsize=0.7\n"
        "    ];\n\n");

    for (size_t i = 0; i < graph->curr_node; i++) {

        const Tensor *t = &graph->nodes[i];

        const char *color;

        switch (t->op_type) {
            case OP_NONE:
                color = "#E5E7EB"; // gray
                break;

            case OP_LINEAR:
                color = "#BFDBFE"; // blue
                break;

            case OP_ADD:
                color = "#BBF7D0"; // green
                break;

            case OP_LAYER_NORM:
                color = "#FDE68A"; // yellow
                break;

            case OP_ATTENTION:
                color = "#DDD6FE"; // purple
                break;

            case OP_GELU:
                color = "#FBCFE8"; // pink
                break;

            default:
                color = "#D1D5DB";
                break;
        }

        char shape_buf[128] = {0};

        if (t->ndim == 0) {
            strcpy(shape_buf, "scalar");
        } else {
            for (size_t d = 0; d < t->ndim; d++) {

                char tmp[32];

                snprintf(tmp,
                         sizeof(tmp),
                         "%zu",
                         t->shape[d]);

                strcat(shape_buf, tmp);

                if (d + 1 < t->ndim)
                    strcat(shape_buf, " × ");
            }
        }

        const char *op = op_name(t->op_type);
        if (!op)
            op = "UNKNOWN";

        fprintf(fp,
            "n%zu [label=<"
            "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"6\">"

            "<TR>"
            "<TD BGCOLOR=\"%s\">"
            "<B>%s</B>"
            "</TD>"
            "</TR>"

            "<TR>"
            "<TD>%s</TD>"
            "</TR>"

            "<TR>"
            "<TD>%s</TD>"
            "</TR>"

            "<TR>"
            "<TD><I>id=%zu</I></TD>"
            "</TR>"

            "</TABLE>"
            ">];\n",
            t->id,
            color,
            t->name,
            op,
            shape_buf,
            t->id
        );
    }

    fprintf(fp, "\n");

    for (size_t i = 0; i < graph->n_nodes; i++) {

        const Tensor *t = &graph->nodes[i];

        for (size_t j = 0; j < t->nsrc; j++) {

            fprintf(fp,
                    "n%zu -> n%zu;\n",
                    t->src[j]->id,
                    t->id);
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);
}

void graph_export_mermaid(const Graph *graph, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return;
    }

    fprintf(fp, "graph LR\n\n");

    printf("Exporting %zu nodes\n", graph->curr_node);

    // Declare nodes
    for (size_t i = 0; i < graph->curr_node; i++) {

        Tensor *t = &graph->nodes[i];

        // Skip invalid/empty tensors
        if (strlen(t->name) == 0) {
            printf("WARNING: node[%zu] has empty name (id=%zu)\n",
                   i,
                   t->id);
            continue;
        }

        fprintf(fp,
                "    n%zu[\"%s\"]\n",
                t->id,
                t->name);
    }

    fprintf(fp, "\n");

    // Declare edges
    for (size_t i = 0; i < graph->curr_node; i++) {

        Tensor *t = &graph->nodes[i];

        if (strlen(t->name) == 0)
            continue;

        for (size_t j = 0; j < t->nsrc; j++) {

            if (!t->src[j])
                continue;

            fprintf(fp,
                    "    n%zu --> n%zu\n",
                    t->src[j]->id,
                    t->id);
        }
    }

    fclose(fp);
}


void graph_export_json(const Graph *graph, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return;
    }

    fprintf(fp, "{\n");

    // Nodes
    fprintf(fp, "  \"nodes\": [\n");

    for (size_t i = 0; i < graph->n_nodes; i++) {

        const Tensor *t = &graph->nodes[i];

        fprintf(fp,
            "    {"
            "\"id\": %zu, "
            "\"name\": \"%s\", "
            "\"op\": \"%s\", "
            "\"dtype\": \"%s\", "
            "\"ndim\": %u, "
            "\"shape\": [",
            t->id,
            t->name,
            op_name(t->op_type),
            dtype_name(t->d_type),
            t->ndim
        );

        for (size_t d = 0; d < t->ndim; d++) {
            fprintf(fp, "%zu", t->shape[d]);

            if (d + 1 < t->ndim)
                fprintf(fp, ", ");
        }

        fprintf(fp, "]}");

        if (i + 1 < graph->n_nodes)
            fprintf(fp, ",");

        fprintf(fp, "\n");
    }

    fprintf(fp, "  ],\n");

    // Edges
    fprintf(fp, "  \"edges\": [\n");

    int first = 1;

    for (size_t i = 0; i < graph->n_nodes; i++) {

        const Tensor *t = &graph->nodes[i];

        for (size_t j = 0; j < t->nsrc; j++) {

            if (!first)
                fprintf(fp, ",\n");

            fprintf(fp,
                "    {\"source\": %zu, \"target\": %zu}",
                t->src[j]->id,
                t->id
            );

            first = 0;
        }
    }

    fprintf(fp, "\n  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
}