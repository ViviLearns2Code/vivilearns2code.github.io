---
layout: post
title:  "Diving Into Controller Runtime"
date:   2021-03-12 17:16:30 +0200
categories: k8s
comments: true
excerpt_separator: <!--more-->
---
In my previous post [Writing Controllers for Kubernetes Custom Resources]() I explored the development process using pure client-go and how it differs from using controller-runtime (and kubebuilder). In this post, I explore how controller-runtime (v0.7.0) uses concepts we know from client-go.

<!--more-->
As described in my previous post, to develop an operator with controller-runtime, we need to
1. define CRD types
2. generate deepcopy functions
3. write reconciler logic
4. create and run the controller
5. perform deployment-related steps (out of scope for this post)
   
I will skip the first two steps as they are similar to development with pure client-go and start with the reconciler logic. Here, controller-runtime needs us to implement the `Reconciler` interface:
```golang
/* source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/reconcile/reconcile.go */
type Reconciler interface {
  Reconcile(context.Context, Request) (Result, error)
}
```
```golang
/* snipped source code from https://github.com/ViviLearns2Code/myoperator/blob/main/controllers/mykind_controller.go */
package controllers
import (
  "k8s.io/apimachinery/pkg/runtime"
  ctrl "sigs.k8s.io/controller-runtime"
  "sigs.k8s.io/controller-runtime/pkg/client"
)

type MyKindReconciler struct {
  client.Client
  Scheme *runtime.Scheme
}

func (r *MyKindReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
  /* snipped reconciler logic */
  return ctrl.Result{}, nil
}
```
After this, we have to create and run the controller. And that's where a new layer of abstraction called the `manager` comes in.

## The Manager
A manager is required to create and start up a controller (there can be multiple controllers associated with a manager). The manager needs a kubeconfig and can be provided with many configuration options. After creating and configuring the manager, we can add our controller to it. Starting the manager also starts all controllers (and other runnables like webhook servers) assigned to it.
```golang
/* snipped source code from https://github.com/ViviLearns2Code/myoperator/blob/main/main.go */
package main

import (
  "k8s.io/apimachinery/pkg/runtime"
  clientgoscheme "k8s.io/client-go/kubernetes/scheme"
  ctrl "sigs.k8s.io/controller-runtime"

  mygroupv1 "myoperator/api/v1"
  "myoperator/controllers"
)

var (
  scheme   = runtime.NewScheme()
)

func init() {
  _ = clientgoscheme.AddToScheme(scheme)
  _ = mygroupv1.AddToScheme(scheme)
}

func main() {
  // create manager
  mgr, _ := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    Scheme:             scheme,
    Port:               9443,
  })

  // create and add controller
  r := &controllers.MyKindReconciler{
    Client: mgr.GetClient(),
    Scheme: mgr.GetScheme(),
  }
  _ = ctrl.NewControllerManagedBy(mgr).
    For(&mygroupv1.MyKind{}).
    Complete(r)

  // start manager
  mgr.Start(ctrl.SetupSignalHandler())
```
There's a lot of magic happening behind these scenes. For example, we might ask ourselves:
* with pure client-go, we generated informers to keep us updated about resources - where are they now?
* with pure client-go, we read from an internal cache (lister) and write to the k8s apiserver (clientset) in our reconciler logic - how does that work now?

The former question is answered by the manager's `cache` component, the latter by its `client` component. Let's look at them next.

## The Manager's Components
A manager's `client` component is responsible for read and write operations in general, while the `cache` can be used to read data from a local index to reduce load on the k8s apiserver. The client component reuses the cache component for some of its read operations.

```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/manager/internal.go */
import (
  "sigs.k8s.io/controller-runtime/pkg/cache"
  "sigs.k8s.io/controller-runtime/pkg/client"
)
type controllerManager struct {
  cache cache.Cache
  client client.Client
  /* ... */
}
```

## The Cache
A cache is a struct that implements the composite `Cache` interface
```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/cache/cache.go */

type Cache interface {
  client.Reader
  Informers
}
type Informers interface {
  GetInformer(ctx context.Context, obj client.Object) (Informer, error)
  GetInformerForKind(ctx context.Context, gvk schema.GroupVersionKind) (Informer, error)
  Start(ctx context.Context) error
  WaitForCacheSync(ctx context.Context) bool
  client.FieldIndexer
}
```
```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/client/interfaces.go */

type Reader interface {
  Get(ctx context.Context, key ObjectKey, obj Object) error
  List(ctx context.Context, list ObjectList, opts ...ListOption) error
}
type FieldIndexer interface {
  IndexField(ctx context.Context, obj Object, field string, extractValue IndexerFunc) error
}
```
The definition of that struct is spread across several files (`pkg/cache/informer_cache.go`, `pkg/cache/internal/deleg_map.go`, `pkg/cache/internal/informers_map.go`, `pkg/cache/internal/cache_reader.go`). If you strip away the details like support for (un)structured types or multiple resources, the struct is more or less a map that maps from objects or GVKs to client-go's `SharedIndexInformer`. The following table how client-go's package `cache` is used to implement controller-runtime's manager cache:

| Function | Description | client-go objects (pkg cache) |
| ------------- |:-------------|:-----|
|`Get` | cached read access | reads from the informer's index `SharedIndexInformer.GetIndexer()` |
|`List` | cached read access | same as above |
|`GetInformer` | retrieves informer for a given runtime object (creates one if doesn't exist) |returns  `SharedIndexInformer` |
|`GetInformerForKind`| same as above, but for GKV | same as above |
|`Start`| runs all informers, meaning it will list & watch the k8s apiserver for resource updates | `ListWatch` |
|`WaitForCacheSync`| waits for all caches to sync | `WaitForCacheSync` |
|`IndexField` | adds field indices over the cache | `SharedIndexInformer.AddIndexers` |

(Note: The functions `GetInformer` and `GetInformerForKind` return an informer that offers functions like `AddEventHandler`, `AddEventHandlerWithResyncPeriod`, `AddIndexers` and `HasSynced`. In case you remember, `AddEventHandler` was the function we used in pure client-go to register handlers that enqueue updates to the controller's workqueue. With controller-runtime, this is taken care of for you.)

## The Client
A client is a struct that implements the composite `Client` interface:
```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/client/interfaces.go */

type Client interface {
  Reader
  Writer
  StatusClient
  Scheme() *runtime.Scheme
  RESTMapper() meta.RESTMapper
}
type Reader interface {
  Get(ctx context.Context, key ObjectKey, obj Object) error
  List(ctx context.Context, list ObjectList, opts ...ListOption) error
}
type Writer interface {
  Create(ctx context.Context, obj Object, opts ...CreateOption) error
  Delete(ctx context.Context, obj Object, opts ...DeleteOption) error
  Update(ctx context.Context, obj Object, opts ...UpdateOption) error
  Patch(ctx context.Context, obj Object, patch Patch, opts ...PatchOption) error
  DeleteAllOf(ctx context.Context, obj Object, opts ...DeleteAllOfOption) error
}
type StatusClient interface {
  Status() StatusWriter
}
```
(Note: `StatusClient` is in charge of updating the status subresource)

The special thing about the manager's client is that it will access the k8s apiserver for write operations and the cache for read operations. The client is what you will use in your reconciler logic to handle resources. Instead of using listers and clientsets separately, you now have a unified interface. If the you want to bypass the cache for read operations, you can use the `ClientDisableCacheFor` option of the manager. 

```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/manager/client_builder.go */

import (
  "sigs.k8s.io/controller-runtime/pkg/cache"
  "sigs.k8s.io/controller-runtime/pkg/client"
)
func (n *newClientBuilder) Build(cache cache.Cache, config *rest.Config, options client.Options) (client.Client, error) {
  c, err := client.New(config, options)
  if err != nil {
    return nil, err
  }

  return client.NewDelegatingClient(client.NewDelegatingClientInput{
    CacheReader:     cache,
    Client:          c,
    UncachedObjects: n.uncached,
  })
}
```
```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/client/split.go */

func NewDelegatingClient(in NewDelegatingClientInput) (Client, error) {
  /* ... */  
  return &delegatingClient{
    scheme: in.Client.Scheme(),
    mapper: in.Client.RESTMapper(),
    Reader: &delegatingReader{
      CacheReader:  in.CacheReader,
      ClientReader: in.Client,
      scheme:       in.Client.Scheme(),
      uncachedGVKs: uncachedGVKs,
    },
    Writer:       in.Client,
    StatusClient: in.Client,
  }, nil
}
```

## And finally, the Controller
It's time to put everything together. [Remember the magical lines for controller creation](#the-manager)? Let's go through them:

```golang
func main() {
  /* ... */
  // create and add controller
  r := &controllers.MyKindReconciler{
    Client: mgr.GetClient(),
    Scheme: mgr.GetScheme(),
  }
  /* ... */
```
Here we reference the manager's client so that we can use it for read/write operations in our reconciler logic.

```golang
func main(){
  /* ... */
  _ = ctrl.NewControllerManagedBy(mgr).
    For(&mygroupv1.MyKind{}).
    Complete(r)

  // start manager
  mgr.Start(ctrl.SetupSignalHandler())
}
```
This is where most of the magic happens. When we tell controller-runtime to build our controller with `NewControllerManagedBy(...).For(...).Complete(...)`, it creates a controller struct with references to our reconciler implementation and a new workqueue for the controller.
  
```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/controller/controller.go */

func New(name string, mgr manager.Manager, options Options) (Controller, error) {
  c, err := NewUnmanaged(name, mgr, options)
  if err != nil {
    return nil, err
  }
  return c, mgr.Add(c)
}
func NewUnmanaged(name string, mgr manager.Manager, options Options) (Controller, error) {

  return &controller.Controller{
      Do: options.Reconciler,
      MakeQueue: func() workqueue.RateLimitingInterface {
        return workqueue.NewNamedRateLimitingQueue(options.RateLimiter, name)
      },
    }, nil
}
```
When the controller is added to the manager, the manager's cache is injected into the controller. Why does the controller need the cache, you might ask, if it already has the client? 

When the manager is started, the control loop is started as well. This will create and run informers for our resources, and these are not managed by the controller, but by the manager cache. In addition, the controller workqueue needs to be registered to the informer. Finally, worker goroutines are launched, each of which will process a workqueue item by calling our custom reconciler logic inside `processNextWorkItem`.
```golang
/* snipped source code from https://github.com/kubernetes-sigs/controller-runtime/blob/release-0.7/pkg/internal/controller/controller.go */

func (c *Controller) Start(ctx context.Context) error {
  // create workqueue
  c.Queue = c.MakeQueue()
  defer c.Queue.ShutDown()

  err := func() error {
    // run informers, registers workqueue for resource updates
    for _, watch := range c.startWatches {
      c.Log.Info("Starting EventSource", "source", watch.src)
      if err := watch.src.Start(ctx, watch.handler, c.Queue, watch.predicates...); err != nil {
        return err
      }
    }
    // wait until informers are synced
    for _, watch := range c.startWatches {
      syncingSource, ok := watch.src.(source.SyncingSource)
      if !ok {
        continue
      }
    }

    // launch workers to process resources
    for i := 0; i < c.MaxConcurrentReconciles; i++ {
      go wait.UntilWithContext(ctx, func(ctx context.Context) {
        // calls custom reconciler logic
        for c.processNextWorkItem(ctx) {
        }
      }, c.JitterPeriod)
    }

    c.Started = true
    return nil
  }()
  if err != nil {
    return err
  }

  <-ctx.Done()
  // stop workers
  return nil
```

## In a nutshell
And finally, here is a picture to summarize what we have found out:

![svg]({{"/images/controller-runtime.svg"}})

I hope that this post helped make some things a bit clearer - if you have any feedback, I'm happy to hear them!