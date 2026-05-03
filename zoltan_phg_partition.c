/*
 * zoltan_phg_partition.c
 * =====================
 * Thin CLI wrapper around Zoltan's Parallel HyperGraph (PHG) partitioner.
 *
 * Usage (single MPI process, called as a subprocess from Python):
 *   echo "<input>" | ./zoltan_phg_partition
 *
 * Input format (stdin):
 *   Line 1 : <n_vertices> <n_hyperedges> <k>
 *   Lines 2+: one hyperedge per line — space-separated 0-based vertex indices
 *
 * Output (stdout):
 *   A single integer — the number of cut hyperedges.
 *   On error, prints "ERROR <message>" and exits non-zero.
 *
 * Compile (see Makefile target or build_zoltan_wrapper.sh):
 *   mpicc -O2 -o zoltan_phg_partition zoltan_phg_partition.c \
 *         -I<zoltan_install>/include \
 *         -L<zoltan_install>/lib -lzoltan -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zoltan.h"

/* ── Hypergraph data (global, filled from stdin before Zoltan runs) ── */

static int    g_n_vertices   = 0;
static int    g_n_hyperedges = 0;
static int    g_k            = 0;

/* Vertex global IDs: 0 … n_vertices-1 */
static ZOLTAN_ID_TYPE *g_vtx_gid  = NULL;

/* CSR-style hyperedge connectivity */
/* g_he_ptr[e] .. g_he_ptr[e+1]-1 gives the vertices of hyperedge e */
static int            *g_he_ptr   = NULL;   /* length n_hyperedges+1 */
static ZOLTAN_ID_TYPE *g_he_gid   = NULL;   /* hyperedge global IDs  */
static ZOLTAN_ID_TYPE *g_pins     = NULL;   /* vertex pins           */
static int             g_n_pins   = 0;

/* Partition assignment returned by Zoltan */
static int *g_partition = NULL;

/* ── Zoltan query callbacks ─────────────────────────────────────────── */

static int cb_num_obj(void *data, int *ierr)
{
    *ierr = ZOLTAN_OK;
    return g_n_vertices;
}

static void cb_obj_list(void *data, int ngid, int nlid,
                        ZOLTAN_ID_PTR gids, ZOLTAN_ID_PTR lids,
                        int wgt_dim, float *obj_wgts, int *ierr)
{
    int i;
    for (i = 0; i < g_n_vertices; i++) {
        gids[i] = (ZOLTAN_ID_TYPE)i;
        lids[i] = (ZOLTAN_ID_TYPE)i;
    }
    *ierr = ZOLTAN_OK;
}

static void cb_hg_size_cs(void *data, int *n_lists, int *n_pins,
                           int *format, int *ierr)
{
    *n_lists = g_n_hyperedges;
    *n_pins  = g_n_pins;
    *format  = ZOLTAN_COMPRESSED_EDGE;   /* one row = one hyperedge */
    *ierr    = ZOLTAN_OK;
}

static void cb_hg_cs(void *data, int ngid, int n_lists, int n_pins,
                     int format,
                     ZOLTAN_ID_PTR edgeGID, int *vtxPtr,
                     ZOLTAN_ID_PTR pinGID, int *ierr)
{
    int e, p, idx = 0;
    for (e = 0; e < g_n_hyperedges; e++) {
        edgeGID[e] = (ZOLTAN_ID_TYPE)e;
        vtxPtr[e]  = g_he_ptr[e];
    }
    for (p = 0; p < g_n_pins; p++)
        pinGID[p] = g_pins[p];
    *ierr = ZOLTAN_OK;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    float ver;
    int   rc, i;

    /* ── 1. Init MPI + Zoltan ── */
    MPI_Init(&argc, &argv);
    rc = Zoltan_Initialize(argc, argv, &ver);
    if (rc != ZOLTAN_OK) {
        fprintf(stderr, "ERROR Zoltan_Initialize failed\n");
        MPI_Finalize();
        return 1;
    }

    /* ── 2. Read hypergraph from stdin ── */
    if (scanf("%d %d %d", &g_n_vertices, &g_n_hyperedges, &g_k) != 3) {
        fprintf(stderr, "ERROR could not read header\n");
        MPI_Finalize();
        return 1;
    }

    if (g_n_vertices < 1 || g_n_hyperedges < 1 || g_k < 2) {
        /* degenerate: no edges → cut is 0 */
        printf("0\n");
        MPI_Finalize();
        return 0;
    }

    g_vtx_gid  = (ZOLTAN_ID_TYPE *)malloc(g_n_vertices   * sizeof(ZOLTAN_ID_TYPE));
    g_he_ptr   = (int *)            malloc((g_n_hyperedges + 1) * sizeof(int));
    g_he_gid   = (ZOLTAN_ID_TYPE *)malloc(g_n_hyperedges   * sizeof(ZOLTAN_ID_TYPE));
    g_partition= (int *)            malloc(g_n_vertices    * sizeof(int));

    /* Read edges — first pass to count total pins */
    /* We store lines temporarily as dynamic arrays */
    int   *tmp_edge_sizes = (int *)calloc(g_n_hyperedges, sizeof(int));
    int  **tmp_edges      = (int **)malloc(g_n_hyperedges * sizeof(int *));
    char   line[1 << 20];  /* 1 MB line buffer */

    /* consume remainder of header line */
    fgets(line, sizeof(line), stdin);

    g_n_pins = 0;
    for (i = 0; i < g_n_hyperedges; i++) {
        if (!fgets(line, sizeof(line), stdin)) {
            fprintf(stderr, "ERROR unexpected EOF at edge %d\n", i);
            MPI_Finalize();
            return 1;
        }
        /* parse space-separated vertex ids */
        int cap = 64, cnt = 0;
        tmp_edges[i] = (int *)malloc(cap * sizeof(int));
        char *tok = strtok(line, " \t\n\r");
        while (tok) {
            if (cnt == cap) { cap *= 2; tmp_edges[i] = realloc(tmp_edges[i], cap * sizeof(int)); }
            tmp_edges[i][cnt++] = atoi(tok);
            tok = strtok(NULL, " \t\n\r");
        }
        tmp_edge_sizes[i] = cnt;
        g_n_pins += cnt;
    }

    /* Build CSR */
    g_pins   = (ZOLTAN_ID_TYPE *)malloc(g_n_pins * sizeof(ZOLTAN_ID_TYPE));
    g_he_ptr[0] = 0;
    int pin_idx = 0;
    for (i = 0; i < g_n_hyperedges; i++) {
        int j;
        for (j = 0; j < tmp_edge_sizes[i]; j++)
            g_pins[pin_idx++] = (ZOLTAN_ID_TYPE)tmp_edges[i][j];
        g_he_ptr[i + 1] = pin_idx;
        free(tmp_edges[i]);
    }
    free(tmp_edges);
    free(tmp_edge_sizes);

    /* ── 3. Create Zoltan object and set PHG parameters ── */
    struct Zoltan_Struct *zz = Zoltan_Create(MPI_COMM_WORLD);

    Zoltan_Set_Param(zz, "DEBUG_LEVEL",     "0");
    Zoltan_Set_Param(zz, "LB_METHOD",       "PHG");
    Zoltan_Set_Param(zz, "LB_APPROACH",     "PARTITION");
    Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", "0");   /* set dynamically below */
    Zoltan_Set_Param(zz, "RETURN_LISTS",    "PARTS");
    Zoltan_Set_Param(zz, "PHG_CUT_OBJECTIVE", "HYPEREDGES");

    char kstr[32];
    snprintf(kstr, sizeof(kstr), "%d", g_k);
    Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", kstr);

    /* Register callbacks */
    Zoltan_Set_Num_Obj_Fn   (zz, cb_num_obj,   NULL);
    Zoltan_Set_Obj_List_Fn  (zz, cb_obj_list,  NULL);
    Zoltan_Set_HG_Size_CS_Fn(zz, cb_hg_size_cs, NULL);
    Zoltan_Set_HG_CS_Fn     (zz, cb_hg_cs,      NULL);

    /* ── 4. Run partitioning ── */
    int changes, n_gid_entries, n_lid_entries;
    int n_import, n_export;
    ZOLTAN_ID_PTR import_gids, import_lids, export_gids, export_lids;
    int *import_procs, *import_to_part, *export_procs, *export_to_part;

    rc = Zoltan_LB_Partition(zz, &changes,
                             &n_gid_entries, &n_lid_entries,
                             &n_import, &import_gids, &import_lids,
                             &import_procs, &import_to_part,
                             &n_export,  &export_gids, &export_lids,
                             &export_procs, &export_to_part);

    if (rc != ZOLTAN_OK) {
        fprintf(stderr, "ERROR Zoltan_LB_Partition returned %d\n", rc);
        Zoltan_Destroy(&zz);
        MPI_Finalize();
        return 1;
    }

    /* ── 5. Recover partition assignment for each vertex ── */
    /* All vertices start in part 0 (single MPI rank, no imports) */
    for (i = 0; i < g_n_vertices; i++) g_partition[i] = 0;

    /* exported objects get their new part */
    for (i = 0; i < n_export; i++)
        g_partition[(int)export_lids[i]] = export_to_part[i];

    Zoltan_LB_Free_Part(&import_gids, &import_lids, &import_procs, &import_to_part);
    Zoltan_LB_Free_Part(&export_gids, &export_lids, &export_procs, &export_to_part);

    /* ── 6. Count cut hyperedges ── */
    int cut = 0;
    for (i = 0; i < g_n_hyperedges; i++) {
        int first_part = g_partition[g_pins[g_he_ptr[i]]];
        int j, is_cut = 0;
        for (j = g_he_ptr[i] + 1; j < g_he_ptr[i + 1]; j++) {
            if (g_partition[g_pins[j]] != first_part) { is_cut = 1; break; }
        }
        cut += is_cut;
    }

    printf("%d\n", cut);

    /* ── 7. Clean up ── */
    Zoltan_Destroy(&zz);
    free(g_vtx_gid); free(g_he_ptr); free(g_he_gid);
    free(g_pins); free(g_partition);

    MPI_Finalize();
    return 0;
}
