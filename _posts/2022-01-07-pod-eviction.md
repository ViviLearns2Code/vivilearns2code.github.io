---
layout: post
title:  "Pod Evictions"
date:   2022-01-07 20:23:36 +0200
categories: k8s
comments: true
---
When does a pod get evicted, and when does a pod get OOMKilled?

## Provoking Pod Evictions
Last year I had to investigate a bug where our custom resource controller failed to react in an expected way to an evicted pod. Our automatic tests did not cover the case of evicted pods and when we tried to reproduce the issue, it was very hard to figure out how to do so.

A pod can get [evicted][1] when a node runs out of a resource - this can be CPU, memory or disk space. The kubelet observes the resource usage in a configurable housekeeping-interval which defaults to 10s and triggers the eviction if necessary (you can read the [following section][2] to learn how to view a node's configuration). Each pod, depending on the resource requests and limits set on it, is classified into one of the three categories: `Burstable`, `BestEffort` and `Guaranteed`. When several pods are present, the kubelet evicts the pods in the order `Burstable > BestEffort > Guaranteed`. The prerequisite for the eviction to happen however, is that the kubelet actually gets to observe the resource shortage. For example, if a node is running out of memory too fast, the linux kernel's OOMKiller intervenes, killing the pod with an `OOMKilled` error before the kubelet can react. 

To provoke a pod eviction, I therefore had to write the following logic and run it in a pod with `0Mi` memory request and no memory limit specfied (burstable pod):
```golang
func create1M() string {
    s := "*"
   return strings.Repeat(s, 1024*1024)
}

func min(x, y int) int {
    if x < y {
        return x
    }
    return y
}

// this function exhausts the container host's (node's) memory
func executeMemoryExhaustion() {
    outFree, err := exec.Command("bash", "-c", "free --mebi | grep '^Mem' | awk '{ print $4 }'").Output()
    if err != nil {
        return
    }
    freeMB, err := strconv.Atoi(strings.TrimSpace(string(outFree)))
    if err != nil {
        return
    }

    slowdownBufferMB := min(freeMB, 200)
    stepMB := 500
    var s []*string = make([]*string, 1000)

    for {
        // comparison unit: MB
        x := create1M()
        outFree, err := exec.Command("bash", "-c", "free --mebi | grep '^Mem' | awk '{ print $4 }'").Output()
        if err != nil {
            return
        }
        freeMB, err := strconv.Atoi(strings.TrimSpace(string(outFree)))
        if err != nil {
            return
        }
        if freeMB <= slowdownBufferMB+stepMB {
            // slow down to avoid OOMKiller
            time.Sleep(time.Second * 1)
            s = append(s, &x)
        } else {
            xStep := strings.Repeat(x, stepMB)
            s = append(s, &xStep)
        }
        if freeMB <= 100 {
            //NODE IS TOO STRONK
            return
        }
    }
}
```
It took a lot of trial and error to set the threshold and step size for the memory increase. These can be optimized depending on cluster size, of course, but when the free memory is approaching around 200MB, a slowdown is necessary or else we easily run into `OOMKilled`. Also note that I had to use an array with pointers to MB-strings instead of continually appending to a string. Appending to a string will temporarily double the memory usage due to copies, making the slowdown effect unpredictable.


[1]: https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/
[2]: https://kubernetes.io/docs/tasks/administer-cluster/reconfigure-kubelet/#generate-the-configuration-file