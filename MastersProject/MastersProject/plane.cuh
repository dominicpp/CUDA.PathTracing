//#pragma once
//
//#include "hit.cuh"
//
//class Plane : public Hit
//{
//public:
//	Vec3 color, normVec, anchorP;
//
//	Plane(Vec3 anchorP, Vec3 normVec, Vec3 color) : anchorP(anchorP), normVec(normVec), color(color) {}
//
//	__host__ __device__ virtual bool hit(const Ray& r, float tmin, float tmax, RecordHit& record) const override;
//};
//
//bool Plane::hit(const Ray& r, float tmin, float tmax, RecordHit& record) const
//{
//	float t = (dot(anchorP, normVec) - dot(normVec, r.getOrigin()) / dot(r.getDirection(), normVec));
//	if (t > r.getTmin() && t < r.getTmax())
//	{
//		Vec3 point = r.pointAt(t);
//		return true;
//	}
//	return false;
//}