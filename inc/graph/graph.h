#ifndef __GRAPH_H__
#define __GRAPH_H__


#include "./context.h"
#include "./tensor.h"

#include <string.h>


typedef struct Graph{
    Context *ctx;
    Tensor *nodes;
    size_t n_nodes; 
    size_t curr_node;
} Graph;


Graph graph_init(Context *ctx, size_t n_nodes);
void graph_free(Graph *graph);
Tensor *graph_alloc_node(Graph *graph);
void graph_print(const Graph *graph);
void graph_export_dot(const Graph *graph, const char *filename);
void graph_export_mermaid(const Graph *graph, const char *filename);
void graph_export_json(const Graph *graph, const char *filename);
#endif  
