package tensor

// BackendDataRefCounter is an optional interface implemented by backend-specific
// data stored in RawTensor.backendData. When Clone copies the backendData pointer,
// it calls AddRef through this interface so the backend's reference count stays
// correct and releasing one clone does not destroy GPU data still referenced by
// other clones.
//
// This interface keeps the tensor package GPU-agnostic (ADR-019): the tensor
// package defines the contract; GPU backends implement it without an import cycle.
type BackendDataRefCounter interface {
	AddRef()
}
