#include <pmsis.h>

#include "layer.h"

void app_kickoff(void *args) {
    struct pi_device cl_dev;
    struct pi_cluster_conf cl_conf;
    struct pi_cluster_task cl_task;

    printf("Starting layer execution.\n\n");

    pi_cluster_conf_init(&cl_conf);
    pi_open_from_conf(&cl_dev, &cl_conf);
    if (pi_cluster_open(&cl_dev))
        pmsis_exit(-1);
    pi_cluster_send_task_to_cl(&cl_dev, pi_cluster_task(&cl_task, layer, NULL));
    pi_cluster_close(&cl_dev);

    pmsis_exit(0);
}

int main() {
    return pmsis_kickoff((void *)app_kickoff);
}
