#include "../../inc/graph/tensor.h"
#include "../../inc/graph/graph.h"
#include "../../inc/ops/op_attention.h"
#include "../../inc/ops/op_linear.h"

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>
#include <unistd.h>      /* isatty       */
#include <sys/ioctl.h>   /* TIOCGWINSZ   */

void tensor_create(
    Tensor *out,
    const char *name, 
    const size_t *shape, 
    const uint8_t ndim, 
    Tensor **src,
    const size_t nsrc,
    const DType d_type,
    const OpType op_type,
    void *op_params,
    TensorType kind
){

    static size_t token_curr_id = 0;
    out->id = token_curr_id++;
    strcpy(out->name, name);

    out->data_offset = 0;
    out->nbytes = 1;
    out->scratch_offset = 0;
    out->nbytes_scratch = 0;
    out->nelems = 1;
    out->ndim = ndim;
    for(size_t i = 0; i < ndim; i++){
         out->nbytes *= shape[i];
    }
    out->nbytes *= dtype_size(d_type);
    memset(out->shape, 0, 4);
    memset(out->stride, 0, 4);
    for(size_t i = 0; i < ndim; i++) out->stride[i] = 1;
    for(size_t i = 0; i < ndim-1; i++) out->stride[i] = shape[i+1];

    memcpy(out->shape, shape, ndim * sizeof(size_t));

    for(size_t i = 0; i < ndim; i++){
        out->nelems *= shape[i];
    }
    

    out->nsrc = nsrc;
    for(size_t i = 0; i < 4; i++) out->src[i] = NULL;
    for(size_t i = 0; i < nsrc; i++) out->src[i] = src[i];
    out->d_type = d_type;
    out->op_type = op_type;
    out->op_params = op_params;
    out->kind = kind;

}




#define PREVIEW_N 4      /* values shown from each end */

/* ---- palette (auto-off when not a TTY) -------------------------------- */
static const char *CR="", *CDIM="", *CNAME="", *CID="", *COP="",
                  *CDT="", *CLBL="", *CNUM="", *CTREE="", *CWARN="",
                  *CACC="", *CRULE="";

static void colors_init(void)
{
    static int done = 0; if (done) return; done = 1;
    if (!isatty(STDOUT_FILENO)) return;
    CR="\x1b[0m"; CDIM="\x1b[2m"; CNAME="\x1b[1;36m"; CID="\x1b[2;37m";
    COP="\x1b[1;35m"; CDT="\x1b[1;33m"; CLBL="\x1b[2;37m"; CNUM="\x1b[0;32m";
    CTREE="\x1b[34m"; CWARN="\x1b[2;31m"; CACC="\x1b[36m"; CRULE="\x1b[2;36m";
}

/* ---- terminal width --------------------------------------------------- */
static int g_width = 80;
static void term_init(void)
{
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        g_width = ws.ws_col;
    if (g_width > 120) g_width = 120;
    if (g_width < 60)  g_width = 60;
}

/* visible length: skip ESC…m sequences, count UTF-8 glyphs as width 1 */
static size_t vislen(const char *s)
{
    size_t w = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        if (*p == 0x1b) { while (*p && *p != 'm') p++; if (*p) p++; continue; }
        if ((*p & 0xC0) != 0x80) w++;
        p++;
    }
    return w;
}

static void row(const char *left, const char *right)
{
    size_t lw = vislen(left), rw = vislen(right);
    if (rw == 0) { printf("%s\n", left); return; }
    int pad = (int)g_width - (int)lw - (int)rw;
    if (pad < 1) pad = 1;
    printf("%s%*s%s\n", left, pad, "", right);
}

/* is this tensor's data actually placed in memory?
   (your pipeline uses offset 0 == unset; swap for a real sentinel if you add one) */
static bool tensor_is_placed(const Tensor *t) { return t->data_offset != 0; }

/* which transformer block a tensor belongs to: "h.<N>." -> N, else -1 */
static int tensor_layer(const Tensor *t)
{
    if (t->name[0] == 'h' && t->name[1] == '.') {
        const char *p = t->name + 2;
        if (*p < '0' || *p > '9') return -1;
        int n = 0;
        while (*p >= '0' && *p <= '9') n = n * 10 + (*p++ - '0');
        return n;
    }
    return -1;
}

/* ---- formatters ------------------------------------------------------- */
static void fmt_bytes(size_t n, char *o, size_t c)
{
    if      (n >= (1ull<<30)) snprintf(o,c,"%.2f GB", n/(double)(1ull<<30));
    else if (n >= (1ull<<20)) snprintf(o,c,"%.2f MB", n/(double)(1ull<<20));
    else if (n >= (1ull<<10)) snprintf(o,c,"%.2f KB", n/(double)(1ull<<10));
    else                      snprintf(o,c,"%zu B",   n);
}
static void fmt_dims(const size_t *v, size_t nd, char *o, size_t c)
{
    size_t p = 0; p += snprintf(o+p, c-p, "[");
    for (size_t i=0;i<nd;i++) p += snprintf(o+p,c-p,"%zu%s",v[i],(i+1<nd)?", ":"");
    snprintf(o+p, c-p, "]");
}
static void fmt_commas(size_t n, char *o, size_t c)
{
    char t[32]; int len = snprintf(t,sizeof t,"%zu",n);
    int ol = len + (len-1)/3;
    if ((size_t)ol+1 > c) { snprintf(o,c,"%zu",n); return; }
    o[ol]='\0'; int oi=ol-1, k=0;
    for (int i=len-1;i>=0;i--){ o[oi--]=t[i]; if(++k%3==0 && i>0) o[oi--]=','; }
}

/* bounded-width float: scientific for extreme magnitudes, fixed otherwise.
   collapses -0.0 / +0.0 to a clean "0". */
static void fmt_float(char *o, size_t c, double v)
{
    if (v == 0.0) { snprintf(o, c, "0"); return; }
    double a = fabs(v);
    if (a < 1e-4 || a >= 1e6) snprintf(o, c, "%.3e", v);   /* e.g. 2.963e+29 */
    else                      snprintf(o, c, "%.4g", v);   /* e.g. -0.078    */
}

/* one scalar -> buffer, dtype-aware */
static int fmt_elem(char *o, size_t c, const Tensor *t, const void *base, size_t i)
{
    if (t->d_type == DTYPE_I32)
        return snprintf(o, c, "%s%d%s", CNUM, ((const int *)base)[i], CR);
    char num[32];
    fmt_float(num, sizeof num, ((const float *)base)[i]);
    return snprintf(o, c, "%s%s%s", CNUM, num, CR);
}

/* data preview row: first/last PREVIEW_N values — only when truly placed */
static void data_row(const Context *ctx, const Tensor *t)
{
    if (!ctx || !ctx->mem || t->nelems == 0) return;
    if (!tensor_is_placed(t)) return;   /* no real data yet → no garbage */

    const void *base = (const void *)(ctx->mem + t->data_offset);
    size_t n = t->nelems;
    char buf[768];
    int p = snprintf(buf, sizeof buf, "  %sdata%s   ", CLBL, CR);

    if (n <= 2 * PREVIEW_N) {
        for (size_t i = 0; i < n && p < (int)sizeof buf; i++) {
            p += fmt_elem(buf+p, sizeof buf-p, t, base, i);
            if (i + 1 < n) p += snprintf(buf+p, sizeof buf-p, ", ");
        }
    } else {
        for (size_t i = 0; i < PREVIEW_N; i++) {
            p += fmt_elem(buf+p, sizeof buf-p, t, base, i);
            p += snprintf(buf+p, sizeof buf-p, ", ");
        }
        p += snprintf(buf+p, sizeof buf-p, "%s…%s ", CDIM, CR);
        for (size_t i = n - PREVIEW_N; i < n; i++) {
            p += fmt_elem(buf+p, sizeof buf-p, t, base, i);
            if (i + 1 < n) p += snprintf(buf+p, sizeof buf-p, ", ");
        }
    }
    printf("%s\n", buf);
}

/* subtle separator when we cross into a new block (or the prologue) */
static void maybe_layer_rule(const Tensor *t)
{
    static int last = -99;
    int lyr = tensor_layer(t);
    if (lyr == last) return;
    last = lyr;

    char label[32];
    if (lyr < 0) snprintf(label, sizeof label, " EMBEDDINGS / IO ");
    else         snprintf(label, sizeof label, " BLOCK %d ", lyr);

    int lab = (int)strlen(label);
    int dashes = g_width - lab;
    if (dashes < 2) dashes = 2;
    int left = dashes / 2, right = dashes - left;

    printf("\n%s", CRULE);
    for (int i=0;i<left;i++) printf("─");
    printf("%s%s%s%s", CDIM, label, CR, CRULE);
    for (int i=0;i<right;i++) printf("─");
    printf("%s\n", CR);
}

/* ---- graph banner ----------------------------------------------------- */
void tensor_print_header(void)
{
    colors_init(); term_init();
    printf("\n%s", CACC);
    for (int i=0;i<g_width;i++) printf("━");
    printf("%s\n  %sCOMPUTATION GRAPH%s\n%s", CR, CNAME, CR, CACC);
    for (int i=0;i<g_width;i++) printf("━");
    printf("%s\n", CR);
}

/* ---- tensor_print (pass NULL ctx to skip the value preview) ----------- */
void tensor_print(const Context *ctx, const Tensor *t)
{
    colors_init(); term_init();
    maybe_layer_rule(t);

    char shape[64], stride[64], nbytes[32], scrbytes[32], nelems[32];
    fmt_dims(t->shape,  t->ndim, shape,  sizeof shape);
    fmt_dims(t->stride, t->ndim, stride, sizeof stride);
    fmt_bytes(t->nbytes,         nbytes,   sizeof nbytes);
    fmt_bytes(t->nbytes_scratch, scrbytes, sizeof scrbytes);
    fmt_commas(t->nelems, nelems, sizeof nelems);

    char L[1024], R[256];

    /* row 1: ● name #id .............................. OP · DTYPE · nD */
    snprintf(L,sizeof L, "%s●%s %s%s%s  %s#%zu%s",
             CACC,CR, CNAME,t->name,CR, CID,t->id,CR);
    snprintf(R,sizeof R, "%s%s%s %s·%s %s%s%s %s·%s %s%uD%s",
             COP,op_name(t->op_type),CR, CDIM,CR,
             CDT,dtype_name(t->d_type),CR, CDIM,CR, CNUM,(unsigned)t->ndim,CR);
    row(L,R);

    /* row 2: shape […] stride […] ................. nelems · nbytes */
    snprintf(L,sizeof L, "  %sshape%s %s%s%s   %sstride%s %s%s%s",
             CLBL,CR, CNUM,shape,CR, CLBL,CR, CNUM,stride,CR);
    snprintf(R,sizeof R, "%s%s%s elems %s·%s %s%s%s",
             CNUM,nelems,CR, CDIM,CR, CNUM,nbytes,CR);
    row(L,R);

    /* row 3: ← src1, src2, ... */
    if (t->nsrc) {
        int p = snprintf(L,sizeof L, "  %s←%s ", CTREE,CR);
        for (size_t i=0;i<t->nsrc && p < (int)sizeof L;i++)
            p += snprintf(L+p, sizeof L-p, "%s%s%s%s",
                          CACC, t->src[i]->name, CR, (i+1<t->nsrc)?", ":"");
        printf("%s\n", L);
    }

    /* row 4: params (inline) */
    if (t->op_params) {
        switch (t->op_type) {
        case OP_ATTENTION: {
            const AttentionParams *p = t->op_params;
            double sc = (p->head_dim>0) ? 1.0/sqrt((double)p->head_dim) : 0.0;
            snprintf(L,sizeof L,
                "  %sparams%s  embed=%s%zu%s  head=%s%zu%s  heads=%s%zu%s  ctx=%s%zu%s  scale=%s%.4f%s",
                CLBL,CR, CNUM,p->embed_dim,CR, CNUM,p->head_dim,CR,
                CNUM,p->n_heads,CR, CNUM,p->ctx_win,CR, CNUM,sc,CR);
            printf("%s\n", L); break;
        }
        case OP_LINEAR: {
            const LinearParams *p = t->op_params;
            snprintf(L,sizeof L,
                "  %sparams%s  trans_weight=%s%s%s  is_bias=%s%s%s",
                CLBL,CR, CNUM,p->trans_weight?"true":"false",CR,
                CNUM,p->is_bias?"true":"false",CR);
            printf("%s\n", L); break;
        }
        default: break;
        }
    }

    /* row 5: mem  data@.. scratch@.. ................. scr_bytes */
    snprintf(L, sizeof L,
        "  %smem%s    data @%s%zu%s%s%s%s   scratch @%s%zu%s%s%s%s",
        CLBL, CR,
        CNUM, t->data_offset, CR,
        CWARN, t->data_offset == 0 ? " (unset)" : "", CR,
        CNUM, t->scratch_offset, CR,
        CWARN, t->scratch_offset == 0 ? " (unset)" : "", CR);
    snprintf(R, sizeof R, "%sscr%s %s%s%s", CLBL, CR, CNUM, scrbytes, CR);
    row(L, R);

    /* row 6: data preview (only when placed + ctx given) */
    data_row(ctx, t);
}