#ifndef __GRAPH_H__
#define __GRAPH_H__


#include "./context.h"
#include "./tensor.h"

#include <string.h>


typedef struct Graph{
    Context ctx;
    
    Tensor *nodes;
    size_t size; 
    size_t capacity;
} Graph;


Graph graph_init(size_t capacity);
void graph_free(Graph *graph);
Tensor *graph_alloc_node(Graph *graph);
size_t graph_plan_memory(Graph *graph);
void graph_execute(Graph *graph, size_t no_tokens);
Tensor *graph_find_node(Graph *graph, const char *name);
void graph_load_weights(Graph *graph, const char *filename);
void graph_print(const Graph *graph);
void graph_print_weights(const Graph *graph);
void graph_write(Graph *graph, const char *filename);
void graph_export_dot(const Graph *graph, const char *filename);
void graph_export_mermaid(const Graph *graph, const char *filename);
void graph_export_json(const Graph *graph, const char *filename);
#endif  
