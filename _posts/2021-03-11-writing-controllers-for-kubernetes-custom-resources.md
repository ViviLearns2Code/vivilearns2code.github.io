---
layout: post
title:  "Writing Controllers For Kubernetes Resources"
date:   2021-03-11 13:12:30 +0200
categories: k8s
comments: true
---
# Writing Controllers for Kubernetes Custom Resources
Kubernetes has become omnipresent. Whether you're part of a development team looking to deploy highly available apps and services or part of a data science team looking to run machine learning workloads in a scalable way - Kubernetes is often the platform of choice. The ecosystem around Kubernetes has grown considerably, and one of the example projects based on top of Kubernetes is Kubeflow, a ML platform offering distributed training, workflow orchestration and many other features. I enjoyed working with it and exploring its features a lot and it makes life so much easier for data scientisis who just want to process their data and train their models in python without worrying about the low-level stuff. However, having amazing tools like Kubeflow doesn't eliminate the need for an engineering mindset and a basic curiosity about what happens under the hood of the system. A DevOps knowledge bottleneck within a data science team can be fatal - once something is broken in the cluster, a team full of data scientists will be blocked and at a loss.

I was curious to learn more about Kubernetes. 

One of things I noticed when checking out Kubeflow were Custom Resource Definitions (CRDs) and their respective controllers. For example, when you create a recurring Kubeflow Pipeline job, you actually create a custom resource of type `ScheduledWorkflow` in API group/version `kubeflow.org/v1beta1` (you can see this easily with `kubectl get scheduledworkflow.v1beta1.kubeflow.org`). All changes made to this resource are observed by a controller, which is basically a control loop running on Kubernetes that reacts to these changes.

The combination of a CRD with a controller is often called an operator. An example that is often used to explain operators is the native Kubernetes resource type `ReplicaSet` from the API group/version `apps/v1`. When you define a ReplicaSet in a yaml file like this
```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: myreplicaset
  labels:
    app: myapp
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mycontainer
        image: myimage:v1
  # ...
```
you actually say that you want 3 pods (replicas) running the specified container. The controller for ReplicaSets is [shipped with Kubernetes](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-controller-manager/) and watches for changes of ReplicaSets and Pods. The moment you create this a Replicaset resource with `kubectl apply -f`, the controller spawns 3 pods. If you kill one of the pods, the controller will spawn another one. Basically, it will always check if the current situation in the cluster matches your definition (this is why Kubernetes is known to be declarative instead of imperative) and do whatever is necessary to reach the target state.

Now that we know the purpose of a controller, let's talk about how we can create our own CRD and controller, just like Kubeflow and other frameworks do to extend Kubernetes.

## Creating an Operator
If you're familiar with golang, there are three ways to create an operator: `client-go`, `controller-runtime` and `kubebuilder`. Every one of these libraries/frameworks adds an additional layer of abstraction and relies on the library before. With `client-go` you start from scratch, `controller-runtime` will take care of some things for you and `kubebuilder` will automate as much as possible. 

Does this mean that you always go with `kubebuilder`, since it automates so much? o answer this, it is helpful to understand how the development process differs using the three options.

## Baseline client-go
Since both kubebuilder and controller-runtime use client-go under the hood, it makes sense to use development with pure client-go as our baseline. The following overview that I provide is by no means comprehensive, but it should give you a pretty concrete idea. There are two sources that explain the entire process very well, one being the [sample-controller](https://github.com/kubernetes/sample-controller/blob/master/docs/controller-client-go.md) and the other one being this [code generation walkthrough](https://www.openshift.com/blog/kubernetes-deep-dive-code-generation-customresources). 

So, you want your controller to somehow observe changes to resources of a specific type on the k8s apiserver and retrieve the resource objects for processing. It is not a good solution for your controller to ping the k8s apiserver directly, as that would put too much load on it. Instead, client-go offers a caching mechanism. 

client-go receives notifications from the k8s apiserver whenever an update happens. A so-called informer retrieves the object in question and hands it over to an indexer that will store and index the object inside an internal cache. The informer will also invoke registered event handlers and pass the object to them. The event handlers are used to push the object key into a workqueue. Your controller then pops the key from there and reads the object from the cache for further processing. 

Event handlers and workqueues are part of your custom controller implementation. When you write the processing (reconciling) logic, you can talk to the k8s apiserver directly (e.g. you create a new pod) or you access the cache for read queries. For the former, you need to generate a clientset and for the latter a lister. In addition to that, you also need to generate the informer itself, so it can watch, cache and pass the resource object.

Now that we have a high-level overview, let's walk through the sample-controller files.

### Define your CRD
Remember the [Replicaset definition from the beginning](#Writing-Controllers-for-Kubernetes-Custom-Resources)? Its first two fields `apiVersion` and `kind` specify the group/version and kind (=resource type). They are followed by a `metadata` section with fields such as `name`, `labels`, etc. These fields are shared by all k8s resource types. Finally, there is a `spec` section with fields specific to ReplicaSets only. 

When we define our CRD in golang, we follow the same structure:

* TypeMeta - Group/Version and Kind fields `apiVersion` and `kind`
* ObjectMeta - Standard k8s `metadatafields like `name`, `namespace`
* Spec - Your custom fields
* Status - Optional status subresource with custom fields

From the sample controller's type definitions we can already guess that the resource type `Foo` is a wrapper around the k8s native resource type `Deployment`. 

```golang
/* source code from https://github.com/kubernetes/sample-controller/blob/master/pkg/apis/samplecontroller/v1alpha1/types.go */
package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type Foo struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   FooSpec   `json:"spec"`
	Status FooStatus `json:"status"`
}

type FooSpec struct {
	DeploymentName string `json:"deploymentName"`
	Replicas       *int32 `json:"replicas"`
}

type FooStatus struct {
	AvailableReplicas int32 `json:"availableReplicas"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// FooList is a list of Foo resources
type FooList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Foo `json:"items"`
}
```
But what about those strange-looking comments `// +...` above the type definitions? These comments are actually tags and serve a purpose!

To use client-go, your resource types need to satisfy the `runtime.Object` interface. The interface comprises several deepcopy functions, and you can generate them with a tool called `deepcopy-gen`. To tell the tool for which types it should generate which deepcopy functions, you can include global (package-wide) and local (specified per type) tags in your source code. 

Global tags are included in a file called `doc.go`: https://github.com/kubernetes/sample-controller/blob/b8d9e8c247129e53962d0dcfc08a4e8b47477318/pkg/apis/samplecontroller/v1alpha1/doc.go. For example, the `// +k8s:deepcopy-gen=package` tag tells `deepcopy-gen` to create deepcopy functions for all types in the package `v1alpha1`. 

Local tags are placed before individual types. The tag `// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object` placed above `type Foo struct {...}`  will tell `deepcopy-gen` to generate an additional deepcopy function for `Foo`.

Running `deepcopy-gen` on your package directory will generate a file `zz_generated.deepcopy.go`: https://github.com/kubernetes/sample-controller/blob/master/pkg/apis/samplecontroller/v1alpha1/zz_generated.deepcopy.go

You probably also noticed in the files above that there are other tags unrelated to deepcopy, like `// +groupName=samplecontroller.k8s.io` or `// +genclient`. These are used by `client-gen` to generate the clientset for your resource type:
https://github.com/kubernetes/sample-controller/tree/master/pkg/generated/clientset/versioned 

In addition to deepcopy functions and the clientset, you need to generate the informer (`informer-gen`) and lister (`lister-gen`) for your resource type as well. Many projects have automated all code generation steps in a script: https://github.com/kubernetes/sample-controller/blob/master/hack/update-codegen.sh so you don't need to manually run every generator in the command line. All code generators are port of the [code-generator](https://github.com/kubernetes/code-generator) project.

Lastly, you need to register your golang types: 
* https://github.com/kubernetes/sample-controller/blob/master/pkg/apis/samplecontroller/v1alpha1/register.go 
* https://github.com/kubernetes/sample-controller/blob/master/pkg/apis/samplecontroller/register.go 
By doing so, we link the golang type `Foo` to the Kubernetes resource type `samplecontroller.k8s.io/v1alpha/Foo`. You can read more about it [here](https://book.kubebuilder.io/cronjob-tutorial/gvks.html#err-but-whats-that-scheme-thing).

### Define your controller logic 
If we look at our sample controller, we see that it watches 2 resource types: the native k8s Deployment and the custom Foo. Therefore, clientset, informer and lister for both resource types are required by the controller.
```golang
/* simplified source code from https://github.com/kubernetes/sample-controller/blob/master/controller.go */

import (
  "k8s.io/client-go/tools/cache"
  "k8s.io/client-go/util/workqueue"
  "k8s.io/client-go/kubernetes"
  appslisters "k8s.io/client-go/listers/apps/v1"
  appsinformers "k8s.io/client-go/informers/apps/v1"

  samplescheme "k8s.io/sample-controller/pkg/generated/clientset/versioned/scheme"
  clientset "k8s.io/sample-controller/pkg/generated/clientset/versioned"
  informers "k8s.io/sample-controller/pkg/generated/informers/externalversions/samplecontroller/v1alpha1"
  listers "k8s.io/sample-controller/pkg/generated/listers/samplecontroller/v1alpha1"
)

func NewController(
  kubeclientset kubernetes.Interface,
  sampleclientset clientset.Interface,
  deploymentInformer appsinformers.DeploymentInformer,
  fooInformer informers.FooInformer) *Controller {

  controller := &Controller{
    kubeclientset:     kubeclientset,
    sampleclientset:   sampleclientset,
    deploymentsLister: deploymentInformer.Lister(),
    deploymentsSynced: deploymentInformer.Informer().HasSynced,
    foosLister:        fooInformer.Lister(),
    foosSynced:        fooInformer.Informer().HasSynced,
    workqueue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "Foos"),
  }

  fooInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
    AddFunc: controller.enqueueFoo,
    UpdateFunc: func(old, new interface{}) {
        controller.enqueueFoo(new)
    },
  })

  deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
    AddFunc: controller.handleObject,
    UpdateFunc: func(old, new interface{}) {
        newDepl := new.(*appsv1.Deployment)
        oldDepl := old.(*appsv1.Deployment)
        if newDepl.ResourceVersion == oldDepl.ResourceVersion {
            return
        }
        controller.handleObject(new)
    },
    DeleteFunc: controller.handleObject,
  })

  return controller
}
```
In the last couple of lines, you can see that the event handler `handleObject` (which pushes objects to the `workqueue`) is registered for the informer to invoke using `AddEventHandler()`. Next, we define the control loop and our reconciler logic.
```golang
/* simplified source code from https://github.com/kubernetes/sample-controller/blob/master/controller.go */

func (c *Controller) Run(threadiness int, stopCh <-chan struct{}) error {
  // Wait for the caches to be synced before starting workers
  if ok := cache.WaitForCacheSync(stopCh, c.deploymentsSynced, c.foosSynced); !ok {
    return fmt.Errorf("failed to wait for caches to sync")
  }

  // Launch workers to process Foo resources
  for i := 0; i < threadiness; i++ {
    go wait.Until(c.runWorker, time.Second, stopCh)
  }

  <-stopCh

  return nil
}

func (c *Controller) runWorker() {
  for c.processNextWorkItem() {
  }
}

func (c *Controller) processNextWorkItem() bool {
  // retrieve object key from workqueue
  // call reconciler logic in syncHandler 
  // if an error occured, requeue object key in workqueue
  // if everything was ok, forget from workqueue
}

func (c *Controller) syncHandler(key string) error {
  // get foo resource using foosLister (cache read access)
  // get deployment with the name specified in foo.spec using deploymentsLister
  // create deployment if it doesn't exist using kubeclientset
  // make number of replicas on deployment match number specified by Foo using kubeclientset
}
```
Finally, we're ready to start up the controller from our `main.go`
```golang
/* simplified source code from https://github.com/kubernetes/sample-controller/blob/master/main.go */
package main

import (
  kubeinformers "k8s.io/client-go/informers"
  "k8s.io/client-go/kubernetes"
  "k8s.io/client-go/tools/clientcmd"
  "k8s.io/klog/v2"

  clientset "k8s.io/sample-controller/pkg/generated/clientset/versioned"
  informers "k8s.io/sample-controller/pkg/generated/informers/externalversions"
  "k8s.io/sample-controller/pkg/signals"
)

func main() {
  // snipped: initialize & parse flags

  // set up signals so we handle the first shutdown signal gracefully
  stopCh := signals.SetupSignalHandler()

  // create clients
  // masterURL, kubeconfig parsed from flags
  cfg, err := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
  kubeClient, err := kubernetes.NewForConfig(cfg)
  exampleClient, err := clientset.NewForConfig(cfg)

  // create informers
  kubeInformerFactory := kubeinformers.NewSharedInformerFactory(kubeClient, time.Second*30)
  exampleInformerFactory := informers.NewSharedInformerFactory(exampleClient, time.Second*30)

  // create controller
  controller := NewController(kubeClient, exampleClient,
    kubeInformerFactory.Apps().V1().Deployments(),
    exampleInformerFactory.Samplecontroller().V1alpha1().Foos())

  // run informers
  kubeInformerFactory.Start(stopCh)
  exampleInformerFactory.Start(stopCh)

  // run controller
  if err = controller.Run(2, stopCh); err != nil {
    klog.Fatalf("Error running controller: %s", err.Error())
  }
}
```
Note that these are just the steps from a code-writing point of view. For the controller to run on the cluster, you need to build go binaries from your source code, build docker images, define deployment yaml files, configure certificates, RBAC permissions, and so on. These steps however are not specific to controllers and need to be done for any project you deploy.
## Comparing client-go to controller-runtime
Using pure client-go, you need to
* define CRD types in golang
* generate deepcopy functions
* generate clientset, informer, lister
* create informer, lister, clientset for your controller to use
* write control loop
* write reconciler logic
* create and run your controller
* project deployment steps
  
You can skip some of the above steps by using controller-runtime. The above list is reduced to
* define CRD types in golang
* generate deepcopy functions
* write reconciler logic
* create and run your controller
* project deployment steps

In addition, controller-runtime offers easy ways to include admission and conversion webhooks. It also handles dynamic (unstructured) clients for you. In the above sample-controller, we used golang types to create typed clients. For some rare use cases, typed clients are not the right choice which is where dynamic clients come in. [You can read more about such a use case here](https://ymmt2005.hatenablog.com/entry/2020/04/14/An_example_of_using_dynamic_client_of_k8s.io/client-go). With pure client-go, you will have to write different logic for the two types of clients whereas with controller-runtime, you can write the same code and it will take care of the differences under the hood.

Since you no longer need to generate and create clientsets, informers and listers anymore, you might ask yourself if controller-runtime is still using any of these under the hood. The answer is yes, and if you're interested in the details, checkout my next post [Controller Runtime Under The Hood](2021-03-12-controller-runtime-under-the-hood) 

## Comparing controller-runtime to kubebuilder
Kubebuilder is not a library per se like client-go or controller-runtime - it is more like a framework that will generate an entire project for you. The project comes with generated files (including deepcopy functions) and you only need to fill in your type definitions and reconciler logic. It has its own tags `// +kubebuilder...`, which you can leverage to e.g. configure RBAC permissions or webhooks (under the hood it is using [controller-tools](https://github.com/kubernetes-sigs/controller-tools). The project even includes deployment and service yaml files that can be `kustomize`-d and a Makefile that automates project deployment steps. The list of steps is even further reduced to
* define CRD types in golang
* write reconciler logic
* create and run your controller
