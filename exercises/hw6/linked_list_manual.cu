#include <cstdio>
#include <cstdlib>
// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

struct list_elem {
    int key;
    list_elem *next;
};

template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes) {
    ptr = (T)malloc(num_bytes);
}

__host__ __device__
void print_element(list_elem *list, int ele_num) {
    list_elem *elem = list;
    for (int i = 0; i < ele_num; i++)
        elem = elem->next;
    printf("key = %d\n", elem->key);
}

__global__ void gpu_print_element(list_elem *list, int ele_num) {
    print_element(list, ele_num);
}

const int num_elem = 256 * 256;
const int ele = 256 * 256 - 1;

int main() {
    list_elem *list_base, *list;
    list_elem *d_list_base, *d_list, *d_list_next;

    // Allocate and initialize the linked list on the host
    alloc_bytes(list_base, sizeof(list_elem));
    list = list_base;
    for (int i = 0; i < num_elem; i++) {
        list->key = i;
        if (i != num_elem - 1) {
            alloc_bytes(list->next, sizeof(list_elem));
        } else {
            list->next = nullptr;
        }
        list = list->next;
    }

    // Print a specific element from the list on the host
    print_element(list_base, ele);

    // Allocate memory for the first element on the GPU
    cudaMalloc(&d_list_base, sizeof(list_elem));
    d_list = d_list_base;

    // Copy the linked list structure to the GPU
    list = list_base;  // Start with the base of the host list
    for (int i = 0; i < num_elem; i++) {
        // Copy the current node from host to device
        cudaMemcpy(d_list, list, sizeof(list_elem), cudaMemcpyHostToDevice);
        cudaCheckErrors("Copying list element failed");

        if (list->next != nullptr) {
            // Allocate the next node on the GPU and update the next pointer on the GPU
            cudaMalloc(&d_list_next, sizeof(list_elem));
            cudaMemcpy(&d_list->next, &d_list_next, sizeof(list_elem *), cudaMemcpyHostToDevice);
            cudaCheckErrors("Copying next pointer failed");
        } else {
            // Set the last element's next to nullptr on the GPU
            d_list_next = nullptr;
            cudaMemcpy(&d_list->next, &d_list_next, sizeof(list_elem *), cudaMemcpyHostToDevice);
        }

        // Move to the next node in the list (both on the host and device)
        list = list->next;
        d_list = d_list_next;
    }

    // Test: Print an element from the linked list on the GPU
    gpu_print_element<<<1, 1>>>(d_list_base, ele);
    cudaDeviceSynchronize();
    cudaCheckErrors("cuda error!");

    // Free the host and device memory (optional but good practice)
    list_elem *temp = list_base;
    while (temp != nullptr) {
        list_elem *next_temp = temp->next;
        free(temp);
        temp = next_temp;
    }

    d_list = d_list_base;
    while (d_list != nullptr) {
        list_elem *d_temp;
        cudaMemcpy(&d_temp, &(d_list->next), sizeof(list_elem *), cudaMemcpyDeviceToHost);
        cudaFree(d_list);
        d_list = d_temp;
    }

    return 0;
}
