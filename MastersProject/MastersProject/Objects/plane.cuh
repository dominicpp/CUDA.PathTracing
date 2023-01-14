//#pragma once
//
//#include "../Hit/hit.cuh"
//#include "../Ray/ray.cuh"
//
//class Plane : public Hit
//{
//public:
//	Vec3 m_normalVector, m_anchorPoint;
//	Material* m_material;
//
//	Plane() = default;
//	Plane(Vec3 normalVector, Vec3 anchorPoint, Material* material) : m_anchorPoint(anchorPoint), m_normalVector(normalVector), m_material(material) {}
//
//	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const override;
//};
//
//bool Plane::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const
//{
//	float t = (dotProduct(m_anchorPoint, m_normalVector) - dotProduct(m_normalVector, ray.getOrigin()) / dotProduct(ray.getDirection(), m_normalVector));
//	if (tmin < t && t < tmax)
//	{
//		hit.rayParameter = t;
//		hit.positionHit = ray.pointAt(hit.rayParameter);
//		hit.normalVector = m_normalVector;
//		hit.material = m_material;
//		return true;
//	}
//	return false;
//}