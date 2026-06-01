#include "../../inc/graph/graph.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
Graph graph_init(size_t capacity){
    Graph graph;
    graph.nodes = calloc(capacity, sizeof(Tensor));
    graph.size = 0;
    graph.capacity = capacity;
    return graph;
}

void graph_free(Graph *graph){
    assert(graph->capacity > 0);
    free(graph->nodes);
    graph->capacity = 0;
}

Tensor *graph_alloc_node(Graph *graph){
    assert(graph->size <= graph->capacity);
    Tensor *node = &graph->nodes[graph->size];
    graph->size++;
    return node;
}

size_t graph_plan_memory(Graph *graph){
    size_t data_offset = 0;
    size_t scratch_bytes = 0;
    size_t activation_bytes = 0;
    for(size_t i = 0; i < graph->size; i++){
        graph->nodes[i].data_offset = data_offset;
        if(graph->nodes[i].op_type != OP_NONE)
            scratch_bytes = OpTable[graph->nodes[i].op_type].scratch_bytes();
        activation_bytes = graph->nodes[i].nbytes;
        data_offset += scratch_bytes + activation_bytes;
    } 
    return data_offset;
}


void graph_execute(Graph *graph){
    for(size_t i = 0; i < graph->size; i++){
        if(graph->nodes[i].op_type == OP_NONE) continue;
         OpTable[graph->nodes[i].op_type].forward(&graph->ctx, &graph->nodes[i]);
    }
}

Tensor *graph_find_node(Graph *graph, const char *name){
    for(size_t i = 0; i < graph->size; i++){
        printf("%s | %s\n", name, graph->nodes[i].name);
        if(strcmp(name, graph->nodes[i].name) == 0){
            printf("matched\n");
            exit(1);
            return &graph->nodes[i];
        }
    }
    return NULL;
}

void graph_load_weights(Graph *graph, const char *model_filename, const char *weights_filename){
    FILE *model_f   = fopen(model_filename, "r");
    FILE *weights_f = fopen(weights_filename, "r");
    if(model_f == NULL || weights_f == NULL) {
        perror("Error opening file");
        exit(1);
    }
    char line[512] = {0};
    // char layer_name[40] = {0};
    size_t offsets[2] = {0};
    for(size_t i = 0;  i < graph->size; i++){
        while(fgets(line, sizeof(line), model_f) != NULL){
            size_t pos = 0;

            if(strstr(line, graph->nodes[i].name)){
                memset(offsets, 0, sizeof(offsets));
                char *offsets_str = line + 40;
                char *token = strtok(offsets_str, ",");
                offsets[0] = strtoull(token, NULL, 10);
                token = strtok(NULL, ",");
                offsets[1] = strtoull(token, NULL, 10);
                
                //Read weights to Context
                fseek(weights_f, offsets[0], SEEK_SET);
                fread(&graph->ctx.mem[graph->nodes[i].data_offset], 1, offsets[1] - offsets[0], weights_f);
                //printf("%s | offsets=[%zu, %zu]\n", graph->nodes[i].name, offsets[0], offsets[1]);
                break;
            }
        }
        rewind(model_f);
    }
    fclose(model_f);
    fclose(weights_f);
}

void graph_print_weights(const Graph *graph){
    for(size_t i = 0; i < 10; i++){
        tensor_print_weights(&graph->ctx, &graph->nodes[i]);
    }
}

void graph_print(const Graph *graph){
    tensor_print_header();
    for(size_t i = 0; i < graph->capacity; i++){
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

    for (size_t i = 0; i < graph->size; i++) {

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

    for (size_t i = 0; i < graph->size; i++) {

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

    printf("Exporting %zu nodes\n", graph->size);

    // Declare nodes
    for (size_t i = 0; i < graph->size; i++) {

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
    for (size_t i = 0; i < graph->size; i++) {

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

    for (size_t i = 0; i < graph->size; i++) {

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

        if (i + 1 < graph->size)
            fprintf(fp, ",");

        fprintf(fp, "\n");
    }

    fprintf(fp, "  ],\n");

    // Edges
    fprintf(fp, "  \"edges\": [\n");

    int first = 1;

    for (size_t i = 0; i < graph->size; i++) {

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